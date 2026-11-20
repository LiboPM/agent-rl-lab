#!/usr/bin/env python3
"""
Summarize all 5 models across all 6 benchmarks (1,320 samples total) and generate:
  1. `eval/multi_benchmark_report.json` (structured per-model, per-benchmark metrics)
  2. `eval/multi_benchmark_report.md` (clean, comprehensive Markdown report)
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
        "0 步训练（仅剥离未训练特殊 Token，保留 BPE 子词与 <|endoftext|>）",
    ),
    (
        "model1_pure_rl_step120",
        "Model 1: 纯 RL (Zero-RL GRPO Step 120)",
        "outputs/merged/grpo-rlla4k-Qwen3-4B-Base-rlla-step120",
        "直接从 Base 跑 120 步 VeRL GRPO（4 Epochs，无任何 SFT）",
    ),
    (
        "model2_pure_sft_ep2",
        "Model 2: 纯 SFT (Full 3,920条 × 2 Epochs)",
        "outputs/sft/sft-rlla4k-Qwen3-4B-Base-rlla-ep2",
        "在 RLLA-4k 全量 3,920 条原数据上跑 2 Epochs LLaMA-Factory 全参 SFT（124 步）",
    ),
    (
        "model3_sft_plus_rl_step120",
        "Model 3: 全量 SFT + RL (Full-SFT + GRPO Step 120)",
        "outputs/merged/grpo-rlla4k-Qwen3-4B-SFT-ep2-step120",
        "以 Model 2（2-Epoch 全量 SFT）为起点继续跑 120 步 VeRL GRPO",
    ),
    (
        "model4_coldstart_sft320_ep1",
        "Model 4: 微缩冷启动 SFT (Cold-Start 320条 × 1 Epoch)",
        "outputs/sft/sft-coldstart320-Qwen3-4B-Base-rlla-ep1",
        "用 Model 1 拒绝采样蒸馏的 320 条真推理样本跑 1 Epoch SFT（仅 5 步更新）",
    ),
]

BENCHMARKS = [
    ("all_1320", "全量 6 大测试集总计 (All 6 Test Sets)", 1320),
    ("rlla_all", "1. RLLA-4k 同分布测试集 (In-Domain Total)", 80),
    ("rlla_single_tool", "  ├─ 1a. RLLA 单工具调用 (Single-Tool)", 36),
    ("rlla_multi_tool", "  ├─ 1b. RLLA 多工具并发 (Multi-Tool >=2)", 35),
    ("rlla_no_tool", "  └─ 1c. RLLA 缺参追问/拒答 (No-Tool)", 9),
    ("bfcl_tool_1000", "2-5. BFCL-v3 工具调用总计 (OOD Tool-Call 4 Sets)", 1000),
    ("bfcl_simple", "  ├─ 2. BFCL-v3 Simple (OOD 单工具调用)", 400),
    ("bfcl_multiple", "  ├─ 3. BFCL-v3 Multiple (OOD 多候选选一)", 200),
    ("bfcl_parallel", "  ├─ 4. BFCL-v3 Parallel (OOD 单工具并发调用)", 200),
    ("bfcl_parallel_multiple", "  └─ 5. BFCL-v3 Parallel-Multiple (OOD 多候选+并发)", 200),
    ("bfcl_irrelevance", "6. BFCL-v3 Irrelevance (OOD 无关工具拒答)", 240),
    ("bfcl_all", "2-6. BFCL-v3 分布外全部总计 (OOD All 5 Sets)", 1240),
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

    if sample["is_rlla"]:
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
    dump_dir = PROJECT_ROOT / "eval" / "multi_bench_dumps"

    all_model_reports = []
    for tag, title, path_str, desc in MODELS:
        dump_file = dump_dir / f"{tag}.jsonl"
        if not dump_file.exists():
            raise FileNotFoundError(f"Missing dump file: {dump_file}")
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

    for rep in all_model_reports:
        print("\n" + "=" * 124)
        print(f"★ {rep['model_title']}")
        print(f"  路径: {rep['model_path']} | 说明: {rep['description']}")
        print("-" * 124)
        print(
            f"{'测试集 (Benchmark)':42s} | {'样本数':>5s} | {'总奖励':>7s} | {'格式分':>6s} | "
            f"{'参数分':>7s} | {'工具名%':>7s} | {'严格EM%':>7s} | {'归一化EM%':>8s} | {'真CoT%':>6s} | {'思考字数':>6s}"
        )
        print("-" * 124)
        for k, b_title, _ in BENCHMARKS:
            b = rep["benchmarks"][k]
            print(
                f"{b_title:42s} | {b['count']:5d} | {b['total_reward']:+7.4f} | {b['format_reward']:6.4f} | "
                f"{b['accuracy_reward']:+7.4f} | {b['tool_name_match_pct']:6.2f}% | "
                f"{b['strict_exact_match_pct']:6.2f}% | {b['schema_norm_exact_match_pct']:7.2f}% | "
                f"{b['genuine_cot_pct']:5.1f}% | {b['avg_think_chars']:6.1f}"
            )
    print("\nSaved full JSON report to:", out_json)


if __name__ == "__main__":
    main()
