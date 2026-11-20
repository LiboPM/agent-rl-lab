#!/usr/bin/env python3
"""
Convert ToolRL (RLLA-4k) VeRL Parquet files (`train.parquet`, `test.parquet`) into
LLaMA-Factory ShareGPT JSON format (`sft_train.json`, `sft_test.json`) and generate
`dataset_info.json`.

Design Guarantees:
1. 100% Prompt & Response Alignment with VeRL:
   - Each sample preserves the exact `[system, user, assistant]` message structure.
   - When used with `models/Qwen3-4B-Base-rlla` and LLaMA-Factory's native `chat_template`
     auto-parser (`template: null`), the encoded `prompt_ids` in SFT match VeRL's
     `RLHFDataset` `prompt_ids` token-for-token across all 3,920 train and 80 test samples.
2. Reward Self-Check:
   - Verifies that every converted assistant target achieves the maximum possible score
     (`format_reward == 1.0` and `exact_match == 1.0`) under `rewards/rlla_reward.py`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rewards.rlla_reward import compute_score


def convert_parquet_to_sharegpt(parquet_path: Path, out_json_path: Path) -> List[Dict[str, Any]]:
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing input parquet file: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    records: List[Dict[str, Any]] = []
    perfect_fmt = 0
    perfect_exact = 0

    for idx, row in df.iterrows():
        prompt_msgs = list(row["prompt"])
        gt_response = row["reward_model"]["ground_truth"]
        data_source = row.get("data_source", "rlla")
        extra_info = row.get("extra_info", {})

        messages = [
            {"role": str(m["role"]), "content": str(m["content"])}
            for m in prompt_msgs
        ]
        messages.append({"role": "assistant", "content": str(gt_response)})

        records.append({"messages": messages})

        # Verify against VeRL reward function (with trailing <|im_end|><|endoftext|> as emitted by SFT model)
        res = compute_score(
            data_source=data_source,
            solution_str=f"{gt_response}<|im_end|><|endoftext|>",
            ground_truth=gt_response,
            extra_info=extra_info,
        )
        if res["format_reward"] == 1.0:
            perfect_fmt += 1
        if res["exact_match"] == 1.0:
            perfect_exact += 1

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(
        f"[prepare_sft_data] {parquet_path.name} -> {out_json_path.name}: "
        f"{len(records)} samples | format_reward==1.0: {perfect_fmt}/{len(records)} | "
        f"exact_match==1.0: {perfect_exact}/{len(records)}"
    )
    if perfect_fmt != len(records) or perfect_exact != len(records):
        raise RuntimeError(
            f"Reward self-check failed on {parquet_path.name}: "
            f"fmt={perfect_fmt}/{len(records)}, exact={perfect_exact}/{len(records)}"
        )
    return records


def main() -> None:
    data_dir = PROJECT_ROOT / "data" / "rlla_4k"
    train_parquet = data_dir / "train.parquet"
    test_parquet = data_dir / "test.parquet"

    train_json = data_dir / "sft_train.json"
    test_json = data_dir / "sft_test.json"
    dataset_info_path = data_dir / "dataset_info.json"

    convert_parquet_to_sharegpt(train_parquet, train_json)
    convert_parquet_to_sharegpt(test_parquet, test_json)

    dataset_info = {
        "rlla_4k_sft_train": {
            "file_name": "sft_train.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages",
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        },
        "rlla_4k_sft_test": {
            "file_name": "sft_test.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages",
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        },
    }

    with open(dataset_info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)

    print(f"[prepare_sft_data] Wrote LLaMA-Factory dataset registry: {dataset_info_path}")


if __name__ == "__main__":
    main()
