#!/usr/bin/env python3
"""
Prepare a Zero-RL compatible Qwen3-4B-Base model directory for ToolRL (RLLA-4k).

Why this is needed for Qwen3-4B-Base:
1. In `Qwen3-4B-Base`, token ID 151643 (`<|endoftext|>`) has a trained embedding (`norm=1.2266`),
   whereas added tokens 151644..151668 (`<|im_start|>`, `<|im_end|>`, `<tool_call>`, `</tool_call>`,
   `<tool_response>`, `</tool_response>`, `<think>`, `</think>`) were never trained in the Base
   checkpoint and share the exact same random embedding vector (`norm=0.3574`) with 2,227 unused
   vocabulary entries (e.g. rare Hangul/CJK characters).
2. When those tags are tokenized as single untrained token IDs (151657..151668), the Base model
   cannot distinguish them in the prompt and samples random unused tokens in their place.
3. By removing the untrained added tokens (151644..151668) from `tokenizer.json` and
   `tokenizer_config.json`, tags like `<think>`, `</think>`, `<tool_call>`, `</tool_call>` are
   tokenized into standard pre-trained BPE subwords (`['<th', 'ink', '>']`, `['<', 'tool', '_call', '>']`),
   matching `<response>` / `</response>` and matching `Qwen2.5-Base` in the original ToolRL paper.
4. In `chat_template`, `<|im_end|>` is paired with `<|endoftext|>` (`151643`, the trained EOS token)
   so the Base model emits `eos_token_id=151643` cleanly at the end of its assistant turn.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_MODEL_DIR = Path("/home/ma-user/work/models/Qwen3-4B-Base")
DST_MODEL_DIR = PROJECT_ROOT / "models" / "Qwen3-4B-Base-rlla"
TRAINED_EOS_ID = 151643


def main() -> None:
    if not SRC_MODEL_DIR.exists():
        raise FileNotFoundError(f"Source model directory not found: {SRC_MODEL_DIR}")

    DST_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Symlink all weights and non-modified config files
    for item in SRC_MODEL_DIR.iterdir():
        if item.name in {"tokenizer.json", "tokenizer_config.json"}:
            continue
        dst_item = DST_MODEL_DIR / item.name
        if dst_item.exists() or dst_item.is_symlink():
            dst_item.unlink()
        os.symlink(item.resolve(), dst_item)

    # 1. Patch tokenizer_config.json
    with open(SRC_MODEL_DIR / "tokenizer_config.json", "r", encoding="utf-8") as f:
        tc = json.load(f)

    tc["added_tokens_decoder"] = {
        str(TRAINED_EOS_ID): tc["added_tokens_decoder"][str(TRAINED_EOS_ID)]
    }
    tc["additional_special_tokens"] = []
    if "<|im_end|><|endoftext|>" not in tc.get("chat_template", ""):
        tc["chat_template"] = tc["chat_template"].replace(
            "<|im_end|>", "<|im_end|><|endoftext|>"
        )

    with open(DST_MODEL_DIR / "tokenizer_config.json", "w", encoding="utf-8") as f:
        json.dump(tc, f, indent=2, ensure_ascii=False)

    # 2. Patch tokenizer.json
    with open(SRC_MODEL_DIR / "tokenizer.json", "r", encoding="utf-8") as f:
        tj = json.load(f)

    tj["added_tokens"] = [
        tok for tok in tj.get("added_tokens", []) if tok.get("id") == TRAINED_EOS_ID
    ]

    with open(DST_MODEL_DIR / "tokenizer.json", "w", encoding="utf-8") as f:
        json.dump(tj, f, ensure_ascii=False)

    # 3. Verify tokenization
    tok = AutoTokenizer.from_pretrained(str(DST_MODEL_DIR))
    print(f"Prepared model directory: {DST_MODEL_DIR}")
    for tag in ["<think>", "</think>", "<tool_call>", "</tool_call>", "<response>", "</response>", "<|endoftext|>"]:
        ids = tok.encode(tag, add_special_tokens=False)
        pieces = [tok.decode([i]) for i in ids]
        print(f"  {tag:15s} -> ids={ids} pieces={pieces}")


if __name__ == "__main__":
    main()
