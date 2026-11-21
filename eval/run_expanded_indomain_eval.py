#!/usr/bin/env python3
"""
Evaluate models on the Expanded 500-Sample Strictly In-Domain Test Set
(`data/rlla_4k/test_indomain_500.parquet`, N=500: 250 ToolACE + 250 xLAM; 250 Single-Tool + 250 Multi-Tool).
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

from eval.compare_models import is_schema_normalized_exact
from eval.run_multi_benchmark_eval import BOILERPLATE_THINKS
from rewards.rlla_reward import compute_score, extract_assistant_response


def load_expanded_500_samples() -> List[Dict[str, Any]]:
    df = pd.read_parquet(PROJECT_ROOT / "data" / "rlla_4k" / "test_indomain_500.parquet")
    samples = []
    for idx, row in df.iterrows():
        gt = str(row["reward_model"]["ground_truth"])
        extra = row["extra_info"]
        samples.append(
            {
                "id": f"rlla_exp500_{idx}",
                "source": extra["source"],
                "sub_category": extra["sub_category"],
                "messages": [
                    {"role": str(m["role"]), "content": str(m["content"])}
                    for m in row["prompt"]
                ],
                "ground_truth_str": gt,
            }
        )
    return samples


def score_expanded_predictions(samples: List[Dict[str, Any]], preds_map: Dict[str, str]) -> Dict[str, Any]:
    rows = []
    for s in samples:
        pred = preds_map[s["id"]]
        gt = s["ground_truth_str"]
        res = compute_score("rlla", pred, gt)
        norm_em = is_schema_normalized_exact(gt, pred, res["exact_match"])
        clean = extract_assistant_response(pred)
        if "<think>" in clean and "</think>" in clean:
            th = clean.split("<think>")[1].split("</think>")[0].strip()
            is_real_cot = 0.0 if (th in BOILERPLATE_THINKS or len(th) < 25) else 1.0
        else:
            is_real_cot = 0.0
        rows.append(
            {
                "source": s["source"],
                "sub_category": s["sub_category"],
                "score": float(res["score"]),
                "format_reward": float(res["format_reward"]),
                "tool_name_match": float(res["tool_name_match"]),
                "strict_em": float(res["exact_match"]),
                "norm_em": float(norm_em),
                "genuine_cot": float(is_real_cot),
            }
        )

    def agg(sub_rows: List[Dict[str, Any]]) -> Dict[str, float]:
        return {
            "count": len(sub_rows),
            "mean_reward": round(float(np.mean([r["score"] for r in sub_rows])), 4),
            "format_pct": round(float(np.mean([r["format_reward"] for r in sub_rows]) * 100), 2),
            "tool_name_match_pct": round(float(np.mean([r["tool_name_match"] for r in sub_rows]) * 100), 2),
            "strict_em_pct": round(float(np.mean([r["strict_em"] for r in sub_rows]) * 100), 2),
            "norm_em_pct": round(float(np.mean([r["norm_em"] for r in sub_rows]) * 100), 2),
            "genuine_cot_pct": round(float(np.mean([r["genuine_cot"] for r in sub_rows]) * 100), 2),
        }

    return {
        "all_500": agg(rows),
        "single_tool_250": agg([r for r in rows if r["sub_category"] == "rlla_single_tool"]),
        "multi_tool_250": agg([r for r in rows if r["sub_category"] == "rlla_multi_tool"]),
        "toolace_250": agg([r for r in rows if r["source"] == "toolace_indomain"]),
        "xlam_250": agg([r for r in rows if r["source"] == "xlam_indomain"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-tag", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--tp-size", type=int, default=2)
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / "eval" / "indomain500_dumps"
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_path = out_dir / f"{args.model_tag}.jsonl"

    samples = load_expanded_500_samples()

    if not dump_path.exists():
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

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
                f.write(
                    json.dumps(
                        {
                            "id": s["id"],
                            "source": s["source"],
                            "sub_category": s["sub_category"],
                            "output": out.outputs[0].text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    preds_map = {json.loads(l)["id"]: json.loads(l)["output"] for l in open(dump_path, "r", encoding="utf-8")}
    summary = score_expanded_predictions(samples, preds_map)
    print(f"=== [{args.model_tag}] Expanded In-Domain 500 Results ===")
    for k, v in summary.items():
        print(f"  {k:18s}: Strict EM = {v['strict_em_pct']:6.2f}% | Norm EM = {v['norm_em_pct']:6.2f}% | Reward = {v['mean_reward']:.3f} | CoT = {v['genuine_cot_pct']:5.1f}%")


if __name__ == "__main__":
    main()
