#!/usr/bin/env python3
"""
Summarize all 8 models across all 4 benchmark suites (3,171 evaluation samples total):
  1. RLLA-4k Original In-Domain Test Set (N=80; plus Clean N=64 vs Noisy-GT N=16)
  2. RLLA-4k Expanded Strictly In-Domain Test Set (N=500: 250 Single-Tool + 250 Multi-Tool; 250 ToolACE + 250 xLAM)
  3. BFCL-v3 Standard OOD Benchmark (N=1,240: 1,000 Tool-Call + 240 Irrelevance)
  4. BFCL-v3 Live Real-World OOD Benchmark (N=1,351: Live-Simple 258 + Live-Multiple 1,053 + Live-Parallel 16 + Live-Parallel-Multiple 24)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.compare_models import is_schema_normalized_exact
from eval.run_bfcl_live_eval import load_bfcl_live_samples
from eval.run_expanded_indomain_eval import load_expanded_500_samples, score_expanded_predictions
from eval.run_multi_benchmark_eval import (
    BOILERPLATE_THINKS,
    load_all_eval_samples,
    score_bfcl_sample,
)
from rewards.rlla_reward import compute_score, extract_assistant_response

MODELS = [
    (
        "model0_zeroshot_base",
        "Model 0: Zero-Shot Base (Qwen3-4B-Base-rlla)",
        "models/Qwen3-4B-Base-rlla",
        "0-step base model after stripping untrained special tokens",
    ),
    (
        "model2a_pure_sft_ep1",
        "Model 1a: Pure SFT (1.0 Epoch, Step 62, Min Eval-Loss=0.1582)",
        "outputs/sft/sft-rlla4k-Qwen3-4B-Base-rlla-ep2/checkpoint-62",
        "Full-data SFT selected at minimum validation loss (Epoch 1.0, checkpoint-62)",
    ),
    (
        "model2_pure_sft_ep2",
        "Model 1b: Pure SFT (2.0 Epochs, Step 124, Min Train-Loss=0.0196)",
        "outputs/sft/sft-rlla4k-Qwen3-4B-Base-rlla-ep2",
        "Full-data SFT trained to convergence of training loss (Epoch 2.0, checkpoint-124)",
    ),
    (
        "model3_sft_plus_rl_step120",
        "Model 2a: Full-SFT (2.0 Ep) + GRPO (Step 120, Negative Control)",
        "outputs/merged/grpo-rlla4k-Qwen3-4B-SFT-ep2-step120",
        "GRPO initialized from low-entropy Full-SFT (boilerplate <think>, entropy=0.08)",
    ),
    (
        "model4_coldstart_sft320_ep1",
        "Model 2b: Cold-Start SFT (320 Genuine-CoT, 1.0 Ep, 5 Steps)",
        "outputs/sft/sft-coldstart320-Qwen3-4B-Base-rlla-ep1",
        "Lightweight format & genuine-CoT warm-up (320 rejection-sampled trajectories, 5 steps)",
    ),
    (
        "model6_coldstart_sft_plus_rl_step100",
        "Model 2c: Cold-Start SFT (320) + GRPO (Step 100, Positive Control)",
        "outputs/merged/grpo-rlla4k-v2reward-ColdStartSFT320-step100",
        "GRPO (v2 reward) initialized from entropy-preserving Cold-Start SFT (selected at val_reward peak Step 100)",
    ),
    (
        "model1_pure_rl_step120",
        "Model 3a: Pure RL (Zero-RL GRPO, 4.0 Ep / Step 120, v1 Reward)",
        "outputs/merged/grpo-rlla4k-Qwen3-4B-Base-rlla-step120",
        "Direct GRPO from Base model with v1 reward (selected at val_reward peak Step 120)",
    ),
    (
        "model5_purerl_v2_ep8_step120",
        "Model 3b: Pure RL (Zero-RL GRPO, 8.0 Ep / Step 120, v2 Anti-Hacking Reward)",
        "outputs/merged/grpo-rlla4k-v2reward-PureRL-ep8-step120",
        "Extended 8-Epoch GRPO with v2 anti-hacking reward (selected at val_reward peak Step 120)",
    ),
]

BENCHMARKS = [
    ("all_1320", "All 6 Multi-Bench Sets (1,320)", 1320),
    ("rlla_all", "1. RLLA-4k In-Domain (80)", 80),
    ("rlla_single_tool", "  ├─ 1a. RLLA Single-Tool (36)", 36),
    ("rlla_multi_tool", "  ├─ 1b. RLLA Multi-Tool (35)", 35),
    ("rlla_no_tool", "  └─ 1c. RLLA No-Tool Abstention (9)", 9),
    ("bfcl_tool_1000", "2-5. BFCL-v3 Standard Tool-Call (1,000)", 1000),
    ("bfcl_simple", "  ├─ 2. BFCL Simple (400)", 400),
    ("bfcl_multiple", "  ├─ 3. BFCL Multiple (200)", 200),
    ("bfcl_parallel", "  ├─ 4. BFCL Parallel (200)", 200),
    ("bfcl_parallel_multiple", "  └─ 5. BFCL Parallel-Multiple (200)", 200),
    ("bfcl_irrelevance", "6. BFCL Irrelevance Abstention (240)", 240),
    ("bfcl_all", "2-6. BFCL-v3 All 5 Sets (1,240)", 1240),
]

LIVE_BENCHMARKS = [
    ("bfcl_live_all", "BFCL-v3 Live Total (1,351)", 1351),
    ("bfcl_live_simple", "  ├─ Live Simple (258)", 258),
    ("bfcl_live_multiple", "  ├─ Live Multiple (1,053)", 1053),
    ("bfcl_live_parallel", "  ├─ Live Parallel (16)", 16),
    ("bfcl_live_parallel_multiple", "  └─ Live Parallel-Multiple (24)", 24),
]


def score_single_item(sample: Dict[str, Any], pred_raw: str) -> Dict[str, float]:
    clean = extract_assistant_response(pred_raw)
    if "<think>" in clean and "</think>" in clean:
        th = clean.split("<think>")[1].split("</think>")[0].strip()
        think_len = len(th)
        is_real_cot = 0.0 if (th in BOILERPLATE_THINKS or len(th) < 25) else 1.0
    else:
        think_len = 0
        is_real_cot = 0.0

    if sample.get("is_rlla", False):
        gt = sample["ground_truth_str"]
        res = compute_score("rlla", pred_raw, gt)
        norm_em = is_schema_normalized_exact(gt, pred_raw, res["exact_match"])
        return {
            "score": float(res["score"]),
            "format_reward": float(res["format_reward"]),
            "accuracy_reward": float(res["accuracy_reward"]),
            "tool_name_match": float(res["tool_name_match"]),
            "strict_exact_match": float(res["exact_match"]),
            "schema_norm_exact_match": float(norm_em),
            "is_collapsed": float(res["is_collapsed"]),
            "resp_chars": float(len(clean)),
            "think_chars": float(think_len),
            "genuine_cot": float(is_real_cot),
        }
    else:
        res = score_bfcl_sample(
            pred_raw, sample["ground_truth_str"], sample["bfcl_possible_gt"]
        )
        res["resp_chars"] = float(len(clean))
        res["think_chars"] = float(think_len)
        res["genuine_cot"] = float(is_real_cot)
        return res


def aggregate_metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
    return {
        "count": len(rows),
        "total_reward": round(float(np.mean([r["score"] for r in rows])), 4),
        "format_reward": round(float(np.mean([r["format_reward"] for r in rows])), 4),
        "accuracy_reward": round(float(np.mean([r["accuracy_reward"] for r in rows])), 4),
        "tool_name_match_pct": round(float(np.mean([r["tool_name_match"] for r in rows]) * 100), 2),
        "strict_exact_match_pct": round(
            float(np.mean([r["strict_exact_match"] for r in rows]) * 100), 2
        ),
        "schema_norm_exact_match_pct": round(
            float(np.mean([r["schema_norm_exact_match"] for r in rows]) * 100), 2
        ),
        "collapsed_pct": round(float(np.mean([r["is_collapsed"] for r in rows]) * 100), 2),
        "avg_resp_chars": round(float(np.mean([r["resp_chars"] for r in rows])), 1),
        "avg_think_chars": round(float(np.mean([r["think_chars"] for r in rows])), 1),
        "genuine_cot_pct": round(float(np.mean([r["genuine_cot"] for r in rows]) * 100), 2),
    }


def main() -> None:
    samples = load_all_eval_samples()
    live_samples = load_bfcl_live_samples()
    exp500_samples = load_expanded_500_samples()

    dump_dir = PROJECT_ROOT / "eval" / "multi_bench_dumps"
    live_dir = PROJECT_ROOT / "eval" / "live_bench_dumps"
    exp500_dir = PROJECT_ROOT / "eval" / "indomain500_dumps"

    all_model_reports = []
    for tag, title, path_str, desc in MODELS:
        dump_file = dump_dir / f"{tag}.jsonl"
        live_file = live_dir / f"{tag}.jsonl"
        exp500_file = exp500_dir / f"{tag}.jsonl"

        with open(dump_file, "r", encoding="utf-8") as f:
            preds = [json.loads(line) for line in f]

        buckets: Dict[str, List[Dict[str, float]]] = {k: [] for k, _, _ in BENCHMARKS}
        for s, p in zip(samples, preds):
            m = score_single_item(s, p["output"])
            sub = s["sub_benchmark"]
            buckets["all_1320"].append(m)
            buckets[sub].append(m)
            if s["is_rlla"]:
                buckets["rlla_all"].append(m)
            else:
                buckets["bfcl_all"].append(m)
                if sub != "bfcl_irrelevance":
                    buckets["bfcl_tool_1000"].append(m)

        bench_summary = {
            k: {
                "title": b_title,
                **aggregate_metrics(buckets[k]),
            }
            for k, b_title, _ in BENCHMARKS
        }

        # Live benchmarks
        if live_file.exists():
            with open(live_file, "r", encoding="utf-8") as f:
                live_preds = [json.loads(line) for line in f]
            live_buckets: Dict[str, List[Dict[str, float]]] = {k: [] for k, _, _ in LIVE_BENCHMARKS}
            for s, p in zip(live_samples, live_preds):
                m = score_single_item(s, p["output"])
                live_buckets["bfcl_live_all"].append(m)
                live_buckets[s["sub_benchmark"]].append(m)
            for k, b_title, _ in LIVE_BENCHMARKS:
                bench_summary[k] = {
                    "title": b_title,
                    **aggregate_metrics(live_buckets[k]),
                }

        # Expanded 500 In-Domain
        if exp500_file.exists():
            preds_map = {json.loads(l)["id"]: json.loads(l)["output"] for l in open(exp500_file, "r", encoding="utf-8")}
            exp_summary = score_expanded_predictions(exp500_samples, preds_map)
            bench_summary["indomain_500"] = exp_summary

        all_model_reports.append(
            {
                "model_tag": tag,
                "model_title": title,
                "model_path": path_str,
                "description": desc,
                "benchmarks": bench_summary,
            }
        )

    out_json = PROJECT_ROOT / "eval" / "multi_benchmark_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_model_reports, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 145)
    print(
        f"{'Model':42s} | {'RLLA-80 EM':>12s} | {'InDom-500 EM':>14s} | {'BFCL-1k EM':>14s} | {'BFCL-Live EM':>14s} | {'Irrel-240':>9s} | {'Real CoT%':>9s}"
    )
    print("-" * 145)
    for rep in all_model_reports:
        b = rep["benchmarks"]
        r80 = f"{b['rlla_all']['strict_exact_match_pct']:5.2f}/{b['rlla_all']['schema_norm_exact_match_pct']:5.2f}%"
        i500 = f"{b['indomain_500']['all_500']['strict_em_pct']:5.2f}/{b['indomain_500']['all_500']['norm_em_pct']:5.2f}%"
        b1k = f"{b['bfcl_tool_1000']['strict_exact_match_pct']:5.2f}/{b['bfcl_tool_1000']['schema_norm_exact_match_pct']:5.2f}%"
        blive = f"{b['bfcl_live_all']['strict_exact_match_pct']:5.2f}/{b['bfcl_live_all']['schema_norm_exact_match_pct']:5.2f}%"
        irr = f"{b['bfcl_irrelevance']['strict_exact_match_pct']:6.2f}%"
        cot = f"{b['bfcl_tool_1000']['genuine_cot_pct']:6.2f}%"
        print(
            f"{rep['model_tag']:42s} | {r80:>12s} | {i500:>14s} | {b1k:>14s} | {blive:>14s} | {irr:>9s} | {cot:>9s}"
        )
    print("=" * 145)
    print("Saved unified report to:", out_json)


if __name__ == "__main__":
    main()
