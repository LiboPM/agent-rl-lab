#!/usr/bin/env python3
"""
Multi-Benchmark Evaluation Suite for Agent Tool-Use RL (`agent-rl-lab`).

Evaluates models across 6 test benchmarks (Total = 1,320 samples):
  1. RLLA-4k Test (In-Domain, N=80):
     - RLLA Single-Tool (N=36)
     - RLLA Multi-Tool (N=35)
     - RLLA No-Tool Clarification (N=9)
  2. BFCL-v3 Simple (OOD Single Tool, N=400)
  3. BFCL-v3 Multiple (OOD Multi-Candidate Tool Selection, N=200)
  4. BFCL-v3 Parallel (OOD Parallel Multi-Call, N=200)
  5. BFCL-v3 Parallel-Multiple (OOD Multi-Candidate + Parallel, N=200)
  6. BFCL-v3 Irrelevance (OOD Abstention / No-Tool Rejection, N=240)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rewards.rlla_reward import (
    compute_format_reward,
    compute_score,
    extract_assistant_response,
    match_score,
    parse_tool_calls,
)

BFCL_DATA_DIR = Path(
    "/home/ma-user/work/zxw/gorilla-bfcl-v3/berkeley-function-call-leaderboard/data"
)

BOILERPLATE_THINKS = {
    "I should use the appropriate tool with proper parameters to respond to the user's need.",
    "I should directly respond to the user's need.",
}

RLLA_SYSTEM_TEMPLATE = """You are a helpful multi-turn dialogue assistant capable of leveraging tool calls to solve user tasks and provide structured chat responses.

**Available Tools**
In your response, you can use the following tools:
{tools_block}

**Steps for Each Turn**
1. **Think:** Recall relevant context and analyze the current user goal.
2. **Decide on Tool Usage:** If a tool is needed, specify the tool and its parameters.
3. **Respond Appropriately:** If a response is needed, generate one while maintaining consistency across user queries.

**Output Format**
```plaintext
<think> Your thoughts and reasoning </think>
<tool_call>
{{"name": "Tool name", "parameters": {{"Parameter name": "Parameter content", "... ...": "... ..."}}}}
{{"name": "... ...", "parameters": {{"... ...": "... ...", "... ...": "... ..."}}}}
...
</tool_call>
<response> AI's final response </response>
```

**Important Notes**
1. You must always include the `<think>` field to outline your reasoning. Provide at least one of `<tool_call>` or `<response>`. Decide whether to use `<tool_call>` (possibly multiple times), `<response>`, or both.
2. You can invoke multiple tool calls simultaneously in the `<tool_call>` fields. Each tool call should be a JSON object with a "name" field and an "parameters" field containing a dictionary of parameters. If no parameters are needed, leave the "parameters" field an empty dictionary.
3. Refer to the previous dialogue records in the history, including the user's queries, previous `<tool_call>`, `<response>`, and any tool feedback noted as `<obs>` (if exists)."""


def format_bfcl_tools_to_rlla(functions: List[Dict[str, Any]]) -> str:
    lines = []
    for idx, fn in enumerate(functions, start=1):
        name = fn.get("name", "")
        desc = fn.get("description", "")
        props = fn.get("parameters", {}).get("properties", {})
        required = set(fn.get("parameters", {}).get("required", []))
        rlla_params = {}
        for k, v in props.items():
            p_info = {
                "description": v.get("description", ""),
                "type": v.get("type", "string"),
            }
            if "default" in v:
                p_info["default"] = v["default"]
            elif k not in required:
                p_info["default"] = ""
            rlla_params[k] = p_info
        lines.append(
            f"{idx}. Name: {name}\n"
            f"Description: {desc}\n"
            f"Parameters: {json.dumps(rlla_params, ensure_ascii=False)}"
        )
    return "\n".join(lines)


def load_bfcl_subset(subset: str) -> List[Dict[str, Any]]:
    q_path = BFCL_DATA_DIR / f"BFCL_v3_{subset}.json"
    a_path = BFCL_DATA_DIR / "possible_answer" / f"BFCL_v3_{subset}.json"

    with open(q_path, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]

    answers_map = {}
    if a_path.exists():
        with open(a_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                answers_map[obj["id"]] = obj["ground_truth"]

    samples = []
    for q in questions:
        qid = q["id"]
        user_turns = q["question"][0]
        user_text = "\n".join(m["content"] for m in user_turns)
        sys_prompt = RLLA_SYSTEM_TEMPLATE.format(
            tools_block=format_bfcl_tools_to_rlla(q["function"])
        )
        usr_prompt = f"**Dialogue Records History**\n<user> {user_text} </user>"
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": usr_prompt},
        ]

        if subset == "irrelevance":
            gt_dummy = "<think> Relevant tool not available. </think>\n<response> None of the available tools can answer this request. </response>"
            possible_gt = None
        else:
            possible_gt = answers_map[qid]
            # Build a canonical ground truth string for format checking
            canon_calls = []
            for call_dict in possible_gt:
                fn_name = next(iter(call_dict.keys()))
                param_spec = call_dict[fn_name]
                canon_params = {}
                for pk, pvals in param_spec.items():
                    non_empty = [v for v in pvals if v != "" and v is not None]
                    if non_empty:
                        canon_params[pk] = non_empty[0]
                canon_calls.append(
                    json.dumps({"name": fn_name, "parameters": canon_params}, ensure_ascii=False)
                )
            gt_dummy = (
                "<think> Reasoning </think>\n<tool_call>\n"
                + "\n".join(canon_calls)
                + "\n</tool_call>"
            )

        samples.append(
            {
                "id": qid,
                "benchmark": f"bfcl_{subset}",
                "messages": messages,
                "ground_truth_str": gt_dummy,
                "bfcl_possible_gt": possible_gt,
            }
        )
    return samples


def norm_primitive(v: Any) -> Any:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return float(v)
    if isinstance(v, float):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        if s.lower() in ("true", "false"):
            return s.lower()
        try:
            return float(s)
        except Exception:
            return s.lower().rstrip(".")
    if isinstance(v, list):
        return [norm_primitive(x) for x in v]
    if isinstance(v, dict):
        return {str(k): norm_primitive(val) for k, val in v.items()}
    return v


def val_matches_bfcl_candidates(pred_val: Any, candidates: List[Any]) -> bool:
    for cand in candidates:
        if pred_val == cand:
            return True
        if norm_primitive(pred_val) == norm_primitive(cand):
            return True
        # Handle tuple vs list in BFCL
        if isinstance(cand, tuple) and isinstance(pred_val, list):
            if norm_primitive(list(cand)) == norm_primitive(pred_val):
                return True
    return False


def score_bfcl_sample(
    pred_raw: str,
    gt_dummy: str,
    possible_gt: List[Dict[str, Any]] | None,
) -> Dict[str, float]:
    """Score a BFCL-v3 prediction using both RLLA fine-grained reward and BFCL possible_answer."""
    clean_pred = extract_assistant_response(pred_raw)
    fmt_reward, is_collapsed = compute_format_reward(clean_pred, gt_dummy, max_reward=1.0, min_reward=0.0)

    # Case 1: Irrelevance (no tool call allowed)
    if possible_gt is None:
        if "<tool_call>" not in clean_pred and "</tool_call>" not in clean_pred:
            acc_reward = 0.0
            name_match = 1.0
            strict_em = 1.0 if fmt_reward > 0 else 0.0
            norm_em = 1.0 if len(clean_pred) > 0 else 0.0
        else:
            acc_reward = -3.0
            name_match = 0.0
            strict_em = 0.0
            norm_em = 0.0
        return {
            "score": float(fmt_reward + acc_reward),
            "format_reward": float(fmt_reward),
            "accuracy_reward": float(acc_reward),
            "tool_name_match": float(name_match),
            "strict_exact_match": float(strict_em),
            "schema_norm_exact_match": float(norm_em),
            "is_collapsed": float(1.0 if is_collapsed else 0.0),
        }

    # Case 2: Tool call required
    try:
        if "<tool_call>" not in clean_pred or "</tool_call>" not in clean_pred:
            raise ValueError("Missing <tool_call>")
        pd_block = clean_pred.split("<tool_call>")[1].split("</tool_call>")[0]
        pd_tools = parse_tool_calls(pd_block)
    except Exception:
        return {
            "score": float(fmt_reward - 3.0),
            "format_reward": float(fmt_reward),
            "accuracy_reward": -3.0,
            "tool_name_match": 0.0,
            "strict_exact_match": 0.0,
            "schema_norm_exact_match": 0.0,
            "is_collapsed": 1.0,
        }

    gt_names = [next(iter(c.keys())) for c in possible_gt]
    pd_names = [t["name"] for t in pd_tools]
    name_score = match_score(gt_names, pd_names)

    # Match each GT tool call against unused predicted tool calls
    used_pd = set()
    all_strict_ok = len(pd_tools) == len(possible_gt)
    all_norm_ok = len(pd_tools) == len(possible_gt)
    total_earned = name_score
    local_max = 1.0

    for gt_call in possible_gt:
        gt_name = next(iter(gt_call.keys()))
        param_spec: Dict[str, List[Any]] = gt_call[gt_name]
        required_keys = {
            k for k, vals in param_spec.items() if "" not in vals and None not in vals
        }
        allowed_keys = set(param_spec.keys())
        local_max += 1.0 + len(required_keys)

        best_idx = -1
        best_sub_score = -1.0
        best_is_strict = False
        best_is_norm = False

        for i, pd_tool in enumerate(pd_tools):
            if i in used_pd or pd_tool.get("name") != gt_name:
                continue
            pd_params = pd_tool.get("parameters", {})
            pd_keys = set(pd_params.keys())

            # Key score: required keys must be present, and no unknown keys outside allowed_keys
            req_present = len(required_keys & pd_keys) / len(required_keys) if required_keys else 1.0
            unknown_keys = pd_keys - allowed_keys
            key_score = req_present if not unknown_keys else req_present * 0.5

            # Value score on required keys + provided optional keys
            val_points = 0.0
            strict_call_ok = required_keys.issubset(pd_keys) and not unknown_keys
            norm_call_ok = required_keys.issubset(pd_keys) and not unknown_keys

            for k in required_keys:
                if k in pd_params and val_matches_bfcl_candidates(pd_params[k], param_spec[k]):
                    val_points += 1.0
                    if pd_params[k] not in param_spec[k]:
                        strict_call_ok = False
                else:
                    strict_call_ok = False
                    norm_call_ok = False

            for k in pd_keys - required_keys:
                if k in param_spec:
                    if not val_matches_bfcl_candidates(pd_params[k], param_spec[k]):
                        strict_call_ok = False
                        norm_call_ok = False
                    elif pd_params[k] not in param_spec[k]:
                        strict_call_ok = False

            sub_score = key_score + val_points
            if sub_score > best_sub_score:
                best_sub_score = sub_score
                best_idx = i
                best_is_strict = strict_call_ok
                best_is_norm = norm_call_ok

        if best_idx >= 0:
            used_pd.add(best_idx)
            total_earned += best_sub_score
            if not best_is_strict:
                all_strict_ok = False
            if not best_is_norm:
                all_norm_ok = False
        else:
            all_strict_ok = False
            all_norm_ok = False

    if all_norm_ok and len(pd_tools) == len(possible_gt):
        acc_reward = 3.0
    else:
        acc_reward = 6.0 * (total_earned / local_max) - 3.0

    return {
        "score": float(fmt_reward + acc_reward),
        "format_reward": float(fmt_reward),
        "accuracy_reward": float(acc_reward),
        "tool_name_match": float(name_score),
        "strict_exact_match": float(1.0 if all_strict_ok else 0.0),
        "schema_norm_exact_match": float(1.0 if all_norm_ok else 0.0),
        "is_collapsed": float(1.0 if is_collapsed else 0.0),
    }


def load_all_eval_samples() -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []

    # 1. RLLA-4k Test (80 samples)
    df_rlla = pd.read_parquet(PROJECT_ROOT / "data" / "rlla_4k" / "test.parquet")
    for idx, row in df_rlla.iterrows():
        gt = row["reward_model"]["ground_truth"]
        if "<tool_call>" not in gt:
            sub_cat = "rlla_no_tool"
        else:
            tools = parse_tool_calls(gt.split("<tool_call>")[1].split("</tool_call>")[0])
            sub_cat = "rlla_single_tool" if len(tools) == 1 else "rlla_multi_tool"
        samples.append(
            {
                "id": f"rlla_test_{idx}",
                "benchmark": "rlla_test_all",
                "sub_benchmark": sub_cat,
                "messages": [
                    {"role": str(m["role"]), "content": str(m["content"])}
                    for m in row["prompt"]
                ],
                "ground_truth_str": gt,
                "bfcl_possible_gt": None,
                "is_rlla": True,
            }
        )

    # 2. BFCL-v3 5 Subsets (400 + 200 + 200 + 200 + 240 = 1,240 samples)
    for subset in ["simple", "multiple", "parallel", "parallel_multiple", "irrelevance"]:
        for item in load_bfcl_subset(subset):
            item["sub_benchmark"] = item["benchmark"]
            item["is_rlla"] = False
            samples.append(item)

    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-tag", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--tp-size", type=int, default=2)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    out_dir = PROJECT_ROOT / "eval" / "multi_bench_dumps"
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_path = out_dir / f"{args.model_tag}.jsonl"

    samples = load_all_eval_samples()
    print(f"[{args.model_tag}] Loaded {len(samples)} total evaluation samples across 6 benchmarks.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    prompts = [
        tokenizer.apply_chat_template(s["messages"], tokenize=False, add_generation_prompt=True)
        for s in samples
    ]

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp_size,
        trust_remote_code=True,
        gpu_memory_utilization=0.65,
        max_model_len=6144,
    )
    sp = SamplingParams(
        n=1,
        temperature=0.0,
        max_tokens=1024,
        stop_token_ids=[151643],
    )
    outputs = llm.generate(prompts, sp)

    with open(dump_path, "w", encoding="utf-8") as f:
        for s, out in zip(samples, outputs):
            pred = out.outputs[0].text
            rec = {
                "id": s["id"],
                "benchmark": s["benchmark"],
                "sub_benchmark": s["sub_benchmark"],
                "output": pred,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[{args.model_tag}] Saved all {len(outputs)} predictions to {dump_path}")


if __name__ == "__main__":
    main()
