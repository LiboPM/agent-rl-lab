#!/usr/bin/env python3
"""
Inspect RLLA-4k dataset statistics and run unit tests on rewards/rlla_reward.py.
"""

import os
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rewards.rlla_reward import compute_score


def classify_gt_pattern(gt: str) -> str:
    has_tool = "<tool_call>" in gt
    has_resp = "<response>" in gt
    if has_tool and has_resp:
        return "think + tool_call + response"
    if has_tool:
        return "think + tool_call"
    if has_resp:
        return "think + response (no tool)"
    return "other"


def inspect_split(name: str, path: Path) -> None:
    df = pd.read_parquet(path)
    print(f"\n==================== {name.upper()} ({len(df)} samples) ====================")
    print("Columns     :", df.columns.tolist())
    print("Abilities   :", df["ability"].value_counts().to_dict())

    gts = [r["ground_truth"] for r in df["reward_model"]]
    patterns = pd.Series([classify_gt_pattern(g) for g in gts]).value_counts().to_dict()
    print("GT Patterns :", patterns)

    prompt_chars = [sum(len(m["content"]) for m in p) for p in df["prompt"]]
    gt_chars = [len(g) for g in gts]
    print(
        f"Prompt chars: min={min(prompt_chars)}, median={sorted(prompt_chars)[len(prompt_chars)//2]}, max={max(prompt_chars)}"
    )
    print(
        f"GT chars    : min={min(gt_chars)}, median={sorted(gt_chars)[len(gt_chars)//2]}, max={max(gt_chars)}"
    )

    # Verify reward function on Ground Truth (should achieve maximum score)
    perfect_scores = []
    for row in df.iloc[:20].to_dict(orient="records"):
        gt = row["reward_model"]["ground_truth"]
        res = compute_score(row["data_source"], gt, gt, row.get("extra_info"))
        perfect_scores.append(res["score"])
    print(f"Reward self-check on first 20 GTs: unique scores = {set(perfect_scores)} (4.0 = tool call exact match, 1.0 = no-tool response match)")


def main() -> None:
    data_dir = PROJECT_ROOT / "data" / "rlla_4k"
    for split in ["train", "test"]:
        p = data_dir / f"{split}.parquet"
        if p.exists():
            inspect_split(split, p)
        else:
            print(f"Missing {p}, run: bash scripts/download_rlla4k.sh")


if __name__ == "__main__":
    main()
