#!/usr/bin/env python3
"""
Upgraded Fine-Grained Rule-Based Reward Function (v2) for ToolRL (RLLA-4k) in VeRL.

Fixes the 3 Bottlenecks Diagnosed from the 5-Epoch Pure RL Run:
1. [Fix 1 — Penalize Redundant Optional Parameters]:
   - In v1, outputting extra optional parameters not present in `ground_truth` (e.g. adding
     `"storage": "cloud"`) still received +3.33 ~ +3.50 out of +4.0, giving almost no gradient
     signal to prune redundant keys even though `exact_match` (`gt_tools == pd_tools`) requires
     zero extra keys.
   - In v2, each extra key in `pd_params.keys() - gt_params.keys()` incurs an explicit `-0.75`
     penalty in `score_tool_calls`, strongly incentivizing minimal, exact parameter dicts.
2. [Fix 2 — Two-Tier Type-Aware Value Matching (`+1.0` Exact / `+0.6` Type-Normalized)]:
   - RLLA-4k ground truth occasionally quotes integer/boolean values as strings (`"1"` vs `1`).
   - v2 awards `+1.0` for strict Python `==` match and `+0.6` when primitive-normalized values
     match (`norm_val(pd) == norm_val(gt)`), eliminating destructive false-negative penalties while
     preserving a `+0.4` advantage margin for matching the exact ground-truth type.
3. [Fix 3 — Balanced No-Tool / Clarification Reward (`+2.5` accuracy instead of `0.0`)]:
   - In v1, correctly abstaining on missing-parameter queries (`<response>...</response>`) gave
     `accuracy_reward = 0.0` (total `+1.0`), whereas calling a tool gave up to `+4.0`. This
     asymmetry caused Pure RL to hallucinate dummy parameters on missing-parameter / irrelevance
     queries.
   - In v2, valid no-tool clarification (`format_reward == 1.0` and no `<tool_call>`) receives
     `accuracy_reward = +2.5` (total `+3.5`), balancing abstention against tool invocation.

Note: `exact_match`, `tool_name_match`, and `is_collapsed` keep the EXACT same definitions as v1
so all validation curves in TensorBoard remain 100% comparable.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

from rewards.rlla_reward import (
    compute_format_reward,
    extract_assistant_response,
    match_score,
    parse_tool_calls,
)


def norm_val(v: Any) -> Any:
    """Normalize primitive int/float/bool/str representations for soft type matching."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        s = v.strip()
        if s.lower() in ("true", "false"):
            return s.lower()
        return s
    return v


def score_tool_calls_v2(
    gt_tools: List[Dict[str, Any]],
    pd_tools: List[Dict[str, Any]],
    max_reward: float = 3.0,
    min_reward: float = -3.0,
) -> Tuple[float, float, float]:
    """
    Upgraded fine-grained tool call matching with:
      - Explicit penalty for redundant/hallucinated parameter keys (`extra_keys`)
      - Two-tier value score (`1.0` for strict match, `0.6` for primitive-normalized match)
      - Explicit penalty when predicted tool count differs from ground-truth tool count
    Returns (scaled_reward, name_match_ratio, strict_exact_match).
    """
    if gt_tools == pd_tools:
        return max_reward, 1.0, 1.0

    gt_names = [t["name"] for t in gt_tools]
    pd_names = [t["name"] for t in pd_tools]
    name_score = match_score(gt_names, pd_names)
    score = name_score

    local_max_possible = 1.0
    used_pd_indices = set()

    for gt_tool in gt_tools:
        gt_name = gt_tool["name"]
        gt_params = gt_tool.get("parameters", {})
        local_max_possible += 1.0 + len(gt_params)

        best_match = None
        best_match_score = -1e9
        best_match_index = -1

        for i, pd_tool in enumerate(pd_tools):
            if i in used_pd_indices or pd_tool.get("name") != gt_name:
                continue

            pd_params = pd_tool.get("parameters", {})
            param_key_score = match_score(list(gt_params.keys()), list(pd_params.keys()))

            param_val_score = 0.0
            for k, v in gt_params.items():
                if k in pd_params:
                    if pd_params[k] == v:
                        param_val_score += 1.0
                    elif norm_val(pd_params[k]) == norm_val(v):
                        param_val_score += 0.6

            # Penalize extra optional/hallucinated parameter keys not in ground truth
            extra_keys = set(pd_params.keys()) - set(gt_params.keys())
            extra_key_penalty = 0.75 * len(extra_keys)

            total_score = param_key_score + param_val_score - extra_key_penalty

            if total_score > best_match_score:
                best_match_score = total_score
                best_match = pd_tool
                best_match_index = i

        if best_match is not None:
            used_pd_indices.add(best_match_index)
            score += max(0.0, best_match_score)

    # Penalize extra hallucinated tool calls beyond ground_truth count
    extra_tools = max(0, len(pd_tools) - len(gt_tools))
    if extra_tools > 0:
        score = max(0.0, score - 0.5 * extra_tools)

    ratio = max(0.0, min(1.0, score / local_max_possible))
    # Cap non-exact matches at 2.4 (out of 3.0) so strict exact match (+3.0) always has a clear +0.6 jump
    scaled = min(2.4, (max_reward - min_reward) * ratio + min_reward)
    return scaled, name_score, 0.0


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Dict[str, Any] | None = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Main entrypoint for VeRL RewardManager (v2).
    """
    predict_str = extract_assistant_response(solution_str)

    format_reward, is_collapsed = compute_format_reward(
        predict_str, ground_truth, max_reward=1.0, min_reward=0.0
    )

    tool_max = 3.0
    tool_min = -3.0

    # Case 1: Ground truth requires NO tool call (direct response / clarification)
    if "<tool_call>" not in ground_truth:
        if "<tool_call>" not in predict_str and "</tool_call>" not in predict_str:
            # Reward valid abstention/clarification with +2.5 accuracy (total +3.5 with format)
            accuracy_reward = 2.5 if format_reward > 0 else 0.0
            name_match = 1.0
            exact_match = 1.0 if format_reward > 0 else 0.0
        else:
            accuracy_reward = tool_min
            name_match = 0.0
            exact_match = 0.0
    else:
        # Case 2: Ground truth requires tool call(s)
        try:
            gt_block = ground_truth.split("<tool_call>")[1].split("</tool_call>")[0]
            gt_tools = parse_tool_calls(gt_block)

            if "<tool_call>" not in predict_str or "</tool_call>" not in predict_str:
                raise ValueError("Missing <tool_call> tags in prediction")

            pd_block = predict_str.split("<tool_call>")[1].split("</tool_call>")[0]
            pd_tools = parse_tool_calls(pd_block)

            accuracy_reward, name_match, exact_match = score_tool_calls_v2(
                gt_tools, pd_tools, max_reward=tool_max, min_reward=tool_min
            )
        except Exception:
            accuracy_reward = tool_min
            name_match = 0.0
            exact_match = 0.0
            is_collapsed = True

    total_score = float(format_reward + accuracy_reward)

    if random.randint(1, 128) == 1:
        print(
            f"[RLLA Reward v2] score={total_score:.2f} "
            f"(fmt={format_reward:.1f}, acc={accuracy_reward:.2f}, exact={exact_match}) "
            f"| pred={predict_str[:180]!r}"
        )

    return {
        "score": total_score,
        "format_reward": float(format_reward),
        "accuracy_reward": float(accuracy_reward),
        "tool_name_match": float(name_match),
        "exact_match": float(exact_match),
        "is_collapsed": float(1.0 if is_collapsed else 0.0),
    }
