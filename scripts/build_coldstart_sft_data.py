#!/usr/bin/env python3
"""
Build a Cold-Start SFT dataset (DeepSeek-R1 style Rejection Sampling SFT) for ToolRL (RLLA-4k).

Why Full-Dataset SFT on Original RLLA-4k Ground Truth Stalls Subsequent GRPO:
1. Original RLLA-4k `ground_truth` only has two boilerplate `<think>` templates across all 3,920 rows:
   - "<think> I should use the appropriate tool with proper parameters to respond to the user's need. </think>"
   - "<think> I should directly respond to the user's need. </think>"
   Training 2 epochs on all 3,920 rows overwrites the Base model's genuine multi-step reasoning
   with this 1-sentence boilerplate and collapses policy entropy (`actor/entropy: 0.72 -> 0.08`).
2. Because all 8 rollouts per prompt become nearly identical (`std(r) ~ 0`), GRPO receives zero
   advantage signal on >93% of prompts.

How This Script Fixes It (Cold-Start Rejection Sampling SFT):
1. Uses the merged Zero-RL checkpoint (`outputs/merged/grpo-rlla4k-Qwen3-4B-Base-rlla-step120`)
   to generate candidate reasoning chains `<think>...</think>` on a small candidate pool of
   `rlla_4k/train.parquet`.
2. Filters for trajectories that:
   - Achieve a **perfect reward** (`format_reward == 1.0` and `exact_match == 1.0`), AND
   - Contain **genuine task-specific reasoning** inside `<think>...</think>` (not boilerplate,
     and reasoning length >= 80 chars).
3. Combines the verified genuine `<think>...</think>` with the canonical ground-truth
   `<tool_call>...</tool_call>` (or `<response>...</response>`) to form `sft_coldstart_train.json`
   (default: 320 high-quality cold-start samples, ~8% of the training set) and registers
   `rlla_4k_sft_coldstart` in `data/rlla_4k/dataset_info.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rewards.rlla_reward import compute_score, extract_assistant_response

BOILERPLATE_THINKS = {
    "I should use the appropriate tool with proper parameters to respond to the user's need.",
    "I should directly respond to the user's need.",
}


def extract_valid_think(pred_str: str, min_think_chars: int = 60) -> str | None:
    """Return `<think>...</think>` if it is well-formed and contains genuine reasoning."""
    text = extract_assistant_response(pred_str)
    if text.count("<think>") != 1 or text.count("</think>") != 1 or not text.startswith("<think>"):
        return None
    inner = text.split("<think>")[1].split("</think>")[0].strip()
    if len(inner) < min_think_chars:
        return None
    if inner in BOILERPLATE_THINKS:
        return None
    return f"<think> {inner} </think>"


def replace_gt_think(ground_truth: str, new_think: str) -> str:
    """Replace the boilerplate `<think>...</think>` in ground_truth with `new_think`."""
    after_think = ground_truth.split("</think>", 1)[1]
    return f"{new_think}{after_think}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "merged" / "grpo-rlla4k-Qwen3-4B-Base-rlla-step120"),
    )
    parser.add_argument(
        "--train-parquet",
        type=str,
        default=str(PROJECT_ROOT / "data" / "rlla_4k" / "train.parquet"),
    )
    parser.add_argument(
        "--target-samples",
        type=int,
        default=320,
        help="Number of high-quality cold-start SFT samples to collect (default: 320 = 5 steps/epoch at BS=64)",
    )
    parser.add_argument(
        "--candidate-pool",
        type=int,
        default=900,
        help="Number of training prompts to run rejection sampling on",
    )
    parser.add_argument("--n-samples", type=int, default=4, help="Rollouts per prompt")
    parser.add_argument("--tp-size", type=int, default=2, help="vLLM tensor parallel size")
    args = parser.parse_args()

    df = pd.read_parquet(args.train_parquet)
    # Shuffle deterministically and take candidate pool
    df_shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    candidates = df_shuffled.iloc[: min(args.candidate_pool, len(df_shuffled))].copy()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    prompts_rendered: List[str] = []
    for _, row in candidates.iterrows():
        msgs = list(row["prompt"])
        prompts_rendered.append(
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        )

    print(
        f"[build_coldstart_sft] Running vLLM Rejection Sampling on {len(prompts_rendered)} prompts "
        f"(n={args.n_samples}) using {args.model_path}..."
    )
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp_size,
        trust_remote_code=True,
        gpu_memory_utilization=0.7,
        max_model_len=6144,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(
        n=args.n_samples,
        temperature=0.7,
        top_p=0.95,
        max_tokens=1024,
        stop_token_ids=[151643],
    )
    outputs = llm.generate(prompts_rendered, sampling_params)

    selected_records: List[Dict[str, Any]] = []
    tool_count = 0
    notool_count = 0

    for idx, (out, (_, row)) in enumerate(zip(outputs, candidates.iterrows())):
        prompt_msgs = list(row["prompt"])
        gt = row["reward_model"]["ground_truth"]
        data_source = row.get("data_source", "rlla")
        extra_info = row.get("extra_info", {})
        is_tool = "<tool_call>" in gt

        best_target = None
        for comp in out.outputs:
            pred_text = comp.text
            res = compute_score(
                data_source=data_source,
                solution_str=pred_text,
                ground_truth=gt,
                extra_info=extra_info,
            )
            if res["format_reward"] == 1.0 and res["exact_match"] == 1.0:
                valid_think = extract_valid_think(pred_text)
                if valid_think is not None:
                    best_target = replace_gt_think(gt, valid_think)
                    break

        if best_target is None:
            continue

        # Double-check synthesized target passes compute_score with 100% format & exact match
        chk = compute_score(
            data_source=data_source,
            solution_str=f"{best_target}<|im_end|><|endoftext|>",
            ground_truth=gt,
            extra_info=extra_info,
        )
        if chk["format_reward"] != 1.0 or chk["exact_match"] != 1.0:
            continue

        messages = [{"role": str(m["role"]), "content": str(m["content"])} for m in prompt_msgs]
        messages.append({"role": "assistant", "content": best_target})
        selected_records.append({"messages": messages})
        if is_tool:
            tool_count += 1
        else:
            notool_count += 1

        if len(selected_records) >= args.target_samples:
            break

    data_dir = PROJECT_ROOT / "data" / "rlla_4k"
    out_json = data_dir / "sft_coldstart_train.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(selected_records, f, ensure_ascii=False, indent=2)

    # Register in dataset_info.json
    dataset_info_path = data_dir / "dataset_info.json"
    if dataset_info_path.exists():
        with open(dataset_info_path, "r", encoding="utf-8") as f:
            dataset_info = json.load(f)
    else:
        dataset_info = {}

    dataset_info["rlla_4k_sft_coldstart"] = {
        "file_name": "sft_coldstart_train.json",
        "formatting": "sharegpt",
        "columns": {"messages": "messages"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system",
        },
    }
    with open(dataset_info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)

    print(
        f"\n[build_coldstart_sft] Saved {len(selected_records)} verified cold-start SFT samples "
        f"(tool={tool_count}, no_tool={notool_count}) to {out_json}"
    )
    if selected_records:
        print("\n=== Sample 0 Verified Cold-Start Assistant Target ===")
        print(selected_records[0]["messages"][-1]["content"][:500])


if __name__ == "__main__":
    main()
