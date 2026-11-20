#!/usr/bin/env python3
"""
Evaluate all models on the Clean High-Quality Benchmarks:
  1. RLLA-4k Clean In-Domain Test Set (N=64, after removing the 16 samples with truncated JSON strings
     or schema-contradicting int/bool string quotes; vs Dirty N=16)
  2. BFCL-v3 Standard Clean Tool-Calling Benchmark (N=1,000: Simple 400 + Multiple 200 + Parallel 200 + Parallel-Multiple 200)
  3. BFCL-v3 Live Real-World User Benchmark (N=1,351: Live-Simple 258 + Live-Multiple 1,053 + Live-Parallel 16 + Live-Parallel-Multiple 24)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.run_multi_benchmark_eval import (
    BFCL_DATA_DIR,
    BOILERPLATE_THINKS,
    RLLA_SYSTEM_TEMPLATE,
    format_bfcl_tools_to_rlla,
    score_bfcl_sample,
)
from rewards.rlla_reward import compute_score, extract_assistant_response, parse_tool_calls


def load_bfcl_live_samples() -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for subset in ["live_simple", "live_multiple", "live_parallel", "live_parallel_multiple"]:
        q_path = BFCL_DATA_DIR / f"BFCL_v3_{subset}.json"
        a_path = BFCL_DATA_DIR / "possible_answer" / f"BFCL_v3_{subset}.json"
        with open(q_path, "r", encoding="utf-8") as f:
            questions = [json.loads(line) for line in f]
        answers_map = {}
        with open(a_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                answers_map[obj["id"]] = obj["ground_truth"]

        for q in questions:
            qid = q["id"]
            turns = q["question"][0]
            sys_extra = "\n".join(m["content"] for m in turns if m["role"] == "system").strip()
            user_text = "\n".join(m["content"] for m in turns if m["role"] == "user").strip()

            tools_str = format_bfcl_tools_to_rlla(q["function"])
            sys_prompt = RLLA_SYSTEM_TEMPLATE.format(tools_block=tools_str)
            if sys_extra:
                usr_prompt = f"**Dialogue Records History**\n<user> {sys_extra}\n\n{user_text} </user>"
            else:
                usr_prompt = f"**Dialogue Records History**\n<user> {user_text} </user>"

            possible_gt = answers_map[qid]
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
                    "sub_benchmark": f"bfcl_{subset}",
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": usr_prompt},
                    ],
                    "ground_truth_str": gt_dummy,
                    "bfcl_possible_gt": possible_gt,
                }
            )
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-tag", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--tp-size", type=int, default=2)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    out_dir = PROJECT_ROOT / "eval" / "live_bench_dumps"
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_path = out_dir / f"{args.model_tag}.jsonl"

    samples = load_bfcl_live_samples()
    print(f"[{args.model_tag}] Loaded {len(samples)} BFCL-v3 Live real-world samples.")

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
        max_model_len=8192,
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
            rec = {
                "id": s["id"],
                "sub_benchmark": s["sub_benchmark"],
                "output": out.outputs[0].text,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[{args.model_tag}] Saved {len(outputs)} predictions to {dump_path}")


if __name__ == "__main__":
    main()
