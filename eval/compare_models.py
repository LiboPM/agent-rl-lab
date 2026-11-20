#!/usr/bin/env python3
"""
Evaluate and compare Zero-Shot Base, Pure RL (GRPO), Pure SFT, and SFT+RL on the
ToolRL (RLLA-4k) 80-sample test set (`data/rlla_4k/test.parquet`).

Outputs:
1. Detailed console comparison across overall metrics, task sub-types (`no_tool`, `single_tool`,
   `multi_tool`), schema-type-normalized exact match, and `<think>` reasoning depth.
2. Case-by-case breakdown where Pure RL beats Pure SFT (and vice versa).
3. JSON summary saved to `eval/rlla4k_model_comparison.json`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rewards.rlla_reward import compute_score, extract_assistant_response, parse_tool_calls

BOILERPLATE_THINKS = {
    "I should use the appropriate tool with proper parameters to respond to the user's need.",
    "I should directly respond to the user's need.",
}


def norm_val(v: Any) -> Any:
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


def is_schema_normalized_exact(gt_str: str, pred_str: str, strict_exact: float) -> float:
    """
    In RLLA-4k, several ground-truth tool calls serialize `int` or `boolean` schema parameters
    as strings (e.g. `{"page": "1"}` when schema says `"type": "int"`, or `{"checkSMTP": "True"}`
    when schema says `"type": "boolean"`).
    This helper checks whether the prediction matches the ground truth after normalizing
    primitive int/bool/string representations.
    """
    if strict_exact == 1.0:
        return 1.0
    if "<tool_call>" not in gt_str:
        return strict_exact
    try:
        gt_tools = parse_tool_calls(gt_str.split("<tool_call>")[1].split("</tool_call>")[0])
        pd_tools = parse_tool_calls(
            extract_assistant_response(pred_str).split("<tool_call>")[1].split("</tool_call>")[0]
        )
        if len(gt_tools) != len(pd_tools):
            return 0.0
        for g, p in zip(gt_tools, pd_tools):
            if g["name"] != p["name"] or set(g["parameters"].keys()) != set(p["parameters"].keys()):
                return 0.0
            for k in g["parameters"]:
                if norm_val(g["parameters"][k]) != norm_val(p["parameters"][k]):
                    return 0.0
        return 1.0
    except Exception:
        return 0.0


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def evaluate_dump(df: pd.DataFrame, dump_rows: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    scores, fmts, accs, names, exacts, norm_exacts, collapsed = [], [], [], [], [], [], []
    think_chars, genuine_think_flags = [], []
    by_cat: Dict[str, Dict[str, List[float]]] = {
        "no_tool": {"score": [], "exact": [], "norm_exact": []},
        "single_tool": {"score": [], "exact": [], "norm_exact": []},
        "multi_tool": {"score": [], "exact": [], "norm_exact": []},
    }

    for r, (_, drow) in zip(dump_rows, df.iterrows()):
        gt = drow["reward_model"]["ground_truth"]
        pred = r["output"]
        res = compute_score("rlla", pred, gt)
        n_exact = is_schema_normalized_exact(gt, pred, res["exact_match"])

        scores.append(res["score"])
        fmts.append(res["format_reward"])
        accs.append(res["accuracy_reward"])
        names.append(res["tool_name_match"])
        exacts.append(res["exact_match"])
        norm_exacts.append(n_exact)
        collapsed.append(res["is_collapsed"])

        clean = extract_assistant_response(pred)
        if "<think>" in clean and "</think>" in clean:
            th = clean.split("<think>")[1].split("</think>")[0].strip()
            think_chars.append(len(th))
            genuine_think_flags.append(0.0 if th in BOILERPLATE_THINKS else 1.0)
        else:
            think_chars.append(0)
            genuine_think_flags.append(0.0)

        cat = drow["category"]
        by_cat[cat]["score"].append(res["score"])
        by_cat[cat]["exact"].append(res["exact_match"])
        by_cat[cat]["norm_exact"].append(n_exact)

    return {
        "model": label,
        "total_reward": round(float(np.mean(scores)), 4),
        "format_reward": round(float(np.mean(fmts)), 4),
        "accuracy_reward": round(float(np.mean(accs)), 4),
        "tool_name_match_pct": round(float(np.mean(names) * 100), 2),
        "strict_exact_match_pct": round(float(np.mean(exacts) * 100), 2),
        "schema_norm_exact_match_pct": round(float(np.mean(norm_exacts) * 100), 2),
        "collapsed_pct": round(float(np.mean(collapsed) * 100), 2),
        "avg_think_chars": round(float(np.mean(think_chars)), 1),
        "genuine_reasoning_pct": round(float(np.mean(genuine_think_flags) * 100), 2),
        "categories": {
            cat: {
                "count": len(vals["score"]),
                "mean_score": round(float(np.mean(vals["score"])), 4),
                "strict_exact_pct": round(float(np.mean(vals["exact"]) * 100), 2),
                "schema_norm_exact_pct": round(float(np.mean(vals["norm_exact"]) * 100), 2),
            }
            for cat, vals in by_cat.items()
        },
    }


def main() -> None:
    df = pd.read_parquet(PROJECT_ROOT / "data" / "rlla_4k" / "test.parquet")
    cats = []
    for _, row in df.iterrows():
        gt = row["reward_model"]["ground_truth"]
        if "<tool_call>" not in gt:
            cats.append("no_tool")
        else:
            tools = parse_tool_calls(gt.split("<tool_call>")[1].split("</tool_call>")[0])
            cats.append("single_tool" if len(tools) == 1 else "multi_tool")
    df["category"] = cats

    val_dir = PROJECT_ROOT / "logs" / "val_dumps"
    candidates = [
        ("0. Zero-Shot Base (Qwen3-4B-Base-rlla)", val_dir / "grpo-rlla4k-Qwen3-4B-Base-rlla-n4g8" / "0.jsonl"),
        ("1. Pure RL (Zero-RL GRPO Step 120)", val_dir / "grpo-rlla4k-Qwen3-4B-Base-rlla-n4g8" / "120.jsonl"),
        ("2. Cold-Start SFT (320 Samples, 1 Ep)", val_dir / "sft-coldstart320-Qwen3-4B-Base-rlla-ep1" / "0.jsonl"),
        ("3. Pure SFT (Full 3920 Samples, 2 Ep)", val_dir / "grpo-rlla4k-Qwen3-4B-SFT-ep2-n4g8" / "0.jsonl"),
        ("4. Full-SFT + RL (GRPO Step 50)", val_dir / "grpo-rlla4k-Qwen3-4B-SFT-ep2-n4g8" / "50.jsonl"),
        ("5. Full-SFT + RL (GRPO Step 120)", val_dir / "grpo-rlla4k-Qwen3-4B-SFT-ep2-n4g8" / "120.jsonl"),
    ]

    results = []
    for label, path in candidates:
        if path.exists():
            results.append(evaluate_dump(df, load_jsonl(path), label))

    out_dir = PROJECT_ROOT / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "rlla4k_model_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("=" * 118)
    print(
        f"{'Model':38s} | {'Reward':>7s} | {'Fmt':>6s} | {'Acc':>6s} | "
        f"{'ToolName':>8s} | {'StrictEM':>8s} | {'SchemaEM':>8s} | {'ThinkLen':>8s} | {'RealCoT':>7s}"
    )
    print("-" * 118)
    for r in results:
        print(
            f"{r['model']:38s} | {r['total_reward']:+7.4f} | {r['format_reward']:6.4f} | "
            f"{r['accuracy_reward']:+6.4f} | {r['tool_name_match_pct']:7.2f}% | "
            f"{r['strict_exact_match_pct']:7.2f}% | {r['schema_norm_exact_match_pct']:7.2f}% | "
            f"{r['avg_think_chars']:6.1f}ch | {r['genuine_reasoning_pct']:6.1f}%"
        )
    print("=" * 118)
    print(f"Saved detailed JSON report to: {out_path}")


if __name__ == "__main__":
    main()
