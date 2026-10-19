#!/usr/bin/env python3
"""
Fine-grained Rule-based Reward Function for ToolRL (RLLA-4k) in VeRL.

References:
1. ToolRL: Reward is All Tool Learning Needs (Qian et al., arXiv:2504.13958)
2. Why Multi-Step Tool-Use RL Collapses and How Supervisory Signals Fix It (Hao et al., arXiv:2606.26027)

VeRL entrypoint signature:
    compute_score(data_source, solution_str, ground_truth, extra_info=None) -> dict
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from typing import Any, Dict, List, Tuple


def match_score(list1: List[Any], list2: List[Any]) -> float:
    """Compute Jaccard-like multiset similarity score considering element frequency."""
    if list1 == list2:
        return 1.0
    if not list1 or not list2:
        return 0.0

    count1 = Counter(list1)
    count2 = Counter(list2)

    intersection = sum(min(count1[k], count2[k]) for k in count1.keys() & count2.keys())
    max_possible = len(list1) + len(list2) - intersection
    return intersection / max_possible if max_possible > 0 else 0.0


def extract_assistant_response(solution_str: str) -> str:
    """Strip chat template headers/footers if present; otherwise return raw decoded response."""
    text = solution_str
    if "<|im_start|>assistant" in text:
        text = text.split("<|im_start|>assistant")[-1]
    elif "<|start_header_id|>assistant<|end_header_id|>" in text:
        text = text.split("<|start_header_id|>assistant<|end_header_id|>")[-1]

    if "<|im_end|>" in text:
        text = text.split("<|im_end|>")[0]
    if "<|eot_id|>" in text:
        text = text.split("<|eot_id|>")[0]
    if "<|endoftext|>" in text:
        text = text.split("<|endoftext|>")[0]
    return text.strip()


def compute_format_reward(
    response: str,
    ground_truth: str,
    max_reward: float = 1.0,
    min_reward: float = 0.0,
) -> Tuple[float, bool]:
    """
    Check whether `response` strictly follows the expected XML tag structure:
      - <think>...</think>\n<response>...</response>
      - <think>...</think>\n<tool_call>\n...\n</tool_call>
      - <think>...</think>\n<tool_call>\n...\n</tool_call>\n<response>...</response>
    Returns (format_reward, is_collapsed).
    """
    has_think = (
        response.count("<think>") == 1
        and response.count("</think>") == 1
        and response.startswith("<think>")
    )
    if not has_think:
        return min_reward, True

    gt_has_tool = "<tool_call>" in ground_truth
    gt_has_resp = "<response>" in ground_truth

    if gt_has_resp and not gt_has_tool:
        pattern = r"^<think>.*?</think>\s*<response>.*?</response>$"
        ok = (
            bool(re.search(pattern, response, re.DOTALL))
            and response.count("<response>") == 1
            and response.count("</response>") == 1
            and "<tool_call>" not in response
        )
        return (max_reward if ok else min_reward), (not ok)

    if gt_has_tool and not gt_has_resp:
        pattern = r"^<think>.*?</think>\s*<tool_call>\n.*?\n</tool_call>$"
        ok = (
            bool(re.search(pattern, response, re.DOTALL))
            and response.count("<tool_call>") == 1
            and response.count("</tool_call>") == 1
            and "<response>" not in response
        )
        return (max_reward if ok else min_reward), (not ok)

    if gt_has_tool and gt_has_resp:
        pattern = r"^<think>.*?</think>\s*<tool_call>\n.*?\n</tool_call>\s*<response>.*?</response>$"
        ok = (
            bool(re.search(pattern, response, re.DOTALL))
            and response.count("<tool_call>") == 1
            and response.count("</tool_call>") == 1
            and response.count("<response>") == 1
            and response.count("</response>") == 1
        )
        return (max_reward if ok else min_reward), (not ok)

    pattern = r"^<think>.*?</think>$"
    ok = bool(re.search(pattern, response, re.DOTALL))
    return (max_reward if ok else min_reward), (not ok)


def parse_tool_calls(block_str: str) -> List[Dict[str, Any]]:
    """Parse newline-separated JSON tool calls inside <tool_call>...</tool_call>."""
    lines = [line.strip() for line in block_str.strip().split("\n") if line.strip()]
    tools = []
    for line in lines:
        obj = json.loads(line)
        if not isinstance(obj, dict) or "name" not in obj or "parameters" not in obj:
            raise ValueError(f"Invalid tool call schema: {line}")
        if not isinstance(obj["parameters"], dict):
            raise ValueError(f"'parameters' must be dict: {line}")
        tools.append(obj)
    return tools


def score_tool_calls(
    gt_tools: List[Dict[str, Any]],
    pd_tools: List[Dict[str, Any]],
    max_reward: float = 3.0,
    min_reward: float = -3.0,
) -> Tuple[float, float, float]:
    """
    Fine-grained tool call matching:
      1. Tool name multiset match score
      2. Parameter key match score per matched tool
      3. Parameter value exact match score per matched tool
    Returns (scaled_reward, name_match_ratio, exact_match).
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
        best_match_score = 0.0
        best_match_index = -1

        for i, pd_tool in enumerate(pd_tools):
            if i in used_pd_indices or pd_tool.get("name") != gt_name:
                continue

            pd_params = pd_tool.get("parameters", {})
            param_key_score = match_score(list(gt_params.keys()), list(pd_params.keys()))
            param_val_score = sum(
                1.0 for k, v in gt_params.items() if k in pd_params and pd_params[k] == v
            )
            total_score = param_key_score + param_val_score

            if total_score > best_match_score:
                best_match_score = total_score
                best_match = pd_tool
                best_match_index = i

        if best_match is not None:
            used_pd_indices.add(best_match_index)
            score += best_match_score

    scaled = (max_reward - min_reward) * (score / local_max_possible) + min_reward
    return scaled, name_score, 0.0


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Dict[str, Any] | None = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Main entrypoint for VeRL NaiveRewardManager.
    Score range: [-3.0, 4.0]
      - format_reward: [0.0, 1.0]
      - accuracy_reward: [-3.0, 3.0]
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
            # Did not hallucinate a tool call when parameters were missing
            accuracy_reward = 0.0
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

            accuracy_reward, name_match, exact_match = score_tool_calls(
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
            f"[RLLA Reward Sample] score={total_score:.2f} "
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
