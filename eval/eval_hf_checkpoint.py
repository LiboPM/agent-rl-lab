#!/usr/bin/env python3
"""
Evaluate any HuggingFace model checkpoint on `data/rlla_4k/test.parquet` (80 samples, Greedy)
using vLLM and save a VeRL-compatible validation dump JSONL (`input`, `output`, `score`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rewards.rlla_reward import compute_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--out-jsonl", type=str, required=True)
    parser.add_argument("--tp-size", type=int, default=2)
    args = parser.parse_args()

    df = pd.read_parquet(PROJECT_ROOT / "data" / "rlla_4k" / "test.parquet")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    prompts = [
        tokenizer.apply_chat_template(list(row["prompt"]), tokenize=False, add_generation_prompt=True)
        for _, row in df.iterrows()
    ]

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp_size,
        trust_remote_code=True,
        gpu_memory_utilization=0.6,
        max_model_len=6144,
    )
    sp = SamplingParams(
        n=1,
        temperature=0.0,
        max_tokens=1024,
        stop_token_ids=[151643],
    )
    outputs = llm.generate(prompts, sp)

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for p_str, out, (_, row) in zip(prompts, outputs, df.iterrows()):
            pred = out.outputs[0].text
            gt = row["reward_model"]["ground_truth"]
            res = compute_score(row.get("data_source", "rlla"), pred, gt, row.get("extra_info"))
            rec = {
                "input": p_str,
                "output": pred,
                "gts": gt,
                "score": res["score"],
                "step": 0,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved evaluation dump ({len(outputs)} samples) to {out_path}")


if __name__ == "__main__":
    main()
