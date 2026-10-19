# Changelog

All notable changes, bug fixes, and experimental findings in `agent-rl-lab` are documented in this file with exact `YYYY-MM-DD HH:MM` (UTC+8) timestamps.

---

## [2026-10-19 19:25] — Connected GitHub Remote (`LiboPM/agent-rl-lab`) & Verified `Qwen3-4B-Base-rlla` Training Progression

- **19:01 — Launched Patched 4-Node 32-GPU Run (`grpo-rlla4k-Qwen3-4B-Base-rlla-n4g8_20261019_190110.log`)**:
  - Validation metrics progression (`val-core/rlla/reward/mean@1` / `tool_name_match` / `exact_match`):
    - **Before fix (`17:01` run, `step:0`)**: `reward = -2.6625`, `tool_name_match = 11.25%`, `exact_match = 0.00%`, `response_length = 721.2` tokens.
    - **After fix (`19:01` run, `step:0`)**: `reward = +0.9627`, `tool_name_match = 77.08%`, `exact_match = 43.75%`.
    - **`step:10`**: `reward = +1.2747`, `tool_name_match = 83.44%`, `exact_match = 45.00%`.
    - **`step:20`**: `reward = +1.1436`, `tool_name_match = 80.04%`, `exact_match = 48.75%`.
    - **`step:30`**: `reward = +1.4658`, `tool_name_match = 87.08%`, `exact_match = 52.50%`.
- **19:25 — Linked GitHub Repository (`git@github.com:LiboPM/agent-rl-lab.git`)**:
  - Configured `origin` remote and pushed `main` branch to `https://github.com/LiboPM/agent-rl-lab`.

---

## [2026-10-19 17:27] — Fix `Qwen3-4B-Base` Untrained Special Token Collapse (`Option A`) & Stop Legacy Run

- **17:20 (`kill -INT 83796`) — Gracefully Stopped Unpatched Training Run**:
  - Sent `SIGINT` (`KeyboardInterrupt`) to the old `17:01` run (`grpo-rlla4k-Qwen3-4B-Base-n4g8_20261019_170151.log` at `step:19`) so it could release all 32 GPUs across the 4-node Ray cluster before launching the patched model.
  - Note: The `KeyboardInterrupt` traceback at the end of `grpo-rlla4k-Qwen3-4B-Base-n4g8_20261019_170151.log` is from this intentional stop, not a training crash.
- **17:25 — Created `scripts/prepare_qwen3_base_tokenizer.py` & `models/Qwen3-4B-Base-rlla`**:
  - **Root Cause**: In `Qwen3-4B-Base`, `model.embed_tokens.weight` (tied with `lm_head.weight`) has a trained embedding for token `151643` (`<|endoftext|>`, `norm=1.2266`), whereas added tokens `151644..151668` (`<|im_start|>`, `<|im_end|>`, `<tool_call>`, `</tool_call>`, `<tool_response>`, `</tool_response>`, `<think>`, `</think>`) were never trained in the Base checkpoint and share the exact same initialization vector (`norm=0.3574`, max diff $\le 1.5 \times 10^{-5}$) with **2,227 unused vocabulary tokens** (rare Hangul/CJK/byte characters).
  - **Fix**:
    1. Built `models/Qwen3-4B-Base-rlla` (symlinking weights to `/home/ma-user/work/models/Qwen3-4B-Base` and patching `tokenizer.json` + `tokenizer_config.json`).
    2. Removed untrained added tokens `151644..151668` from `added_tokens` / `added_tokens_decoder`, keeping only `151643` (`<|endoftext|>`).
    3. `<think>`, `</think>`, `<tool_call>`, `</tool_call>` now tokenize into pre-trained BPE subwords (`['<th', 'ink', '>']`, `['</', 'think', '>']`, `['<', 'tool', '_call', '>']`, `['</', 'tool', '_call', '>']`), matching `<response>` / `</response>` and matching `Qwen2.5-Base` in the original ToolRL paper.
    4. Updated `chat_template` turn endings from `<|im_end|>` to `<|im_end|><|endoftext|>` so the Base model naturally emits `eos_token_id=151643` at the end of its assistant turn (~105–215 tokens instead of hitting the 1024-token cap).
- **17:25 — Updated `rewards/rlla_reward.py`**:
  - Updated `extract_assistant_response` to strip trailing `<|im_end|>`, `<|eot_id|>`, and `<|endoftext|>` even when `<|im_start|>assistant` is not present in the decoded response slice.
- **17:26 — Updated `scripts/run_grpo_rlla.sh`**:
  - Switched default `MODEL_PATH` to `${PROJECT_ROOT}/models/Qwen3-4B-Base-rlla` (auto-generating it via `scripts/prepare_qwen3_base_tokenizer.py` if missing).

---

## [2026-10-19 17:12] — Persistent Disk Logging Architecture (`logs/` & `scripts/run_grpo_rlla.sh`)

- **17:12 — Standardized Disk Logging Layout under `/home/ma-user/work/LB/github/agent-rl-lab/logs/`**:
  - `logs/${EXPERIMENT_NAME}_${START_TIME}.log`: Full console & Ray actor stdout/stderr log persisted via `tee`.
  - `logs/latest.log`: Symlink always pointing to the latest active `.log` file.
  - `logs/metrics/${EXPERIMENT_NAME}_${START_TIME}.jsonl`: Structured per-step JSONL scalar metrics via VeRL's `file` logger (`VERL_FILE_LOGGER_PATH`).
  - `logs/tensorboard/${EXPERIMENT_NAME}/`: TensorBoard event files (`TENSORBOARD_DIR`).
  - `logs/val_dumps/${EXPERIMENT_NAME}/`: Validation rollout generations and reward breakdowns (`trainer.validation_data_dir`).

---

## [2026-10-19 17:01] — Fixed `RewardLoopWorker` Custom Reward Config & Launched 4-Node Run

- **16:53 — Diagnosed `NotImplementedError` in `grpo-rlla4k-Qwen3-4B-Base-n4g8_20261019_164829.log`**:
  - `RewardLoopWorker` in `/home/ma-user/work/LB/verl` reads `config.reward.custom_reward_function.path` (`verl/trainer/ppo/reward.py:70`) rather than only the legacy top-level `custom_reward_function.path`.
- **17:01 — Updated `REWARD_ARGS` in `scripts/run_grpo_rlla.sh`**:
  - Set `reward.reward_manager.name=naive`, `reward.custom_reward_function.path="${REWARD_FN_PATH}"`, and `reward.custom_reward_function.name=compute_score` alongside legacy `custom_reward_function.*` keys.
  - Verified 4-node 32-GPU GRPO training (`step:0` validation + `step:1..19` training at ~45s/step, ~1,050 tokens/s/GPU).

---

## [2026-10-19 16:48] — Fixed Multi-Node Ray Worker `PYTHONPATH` Propagation

- **16:43 — Diagnosed `AttributeError` in `grpo-rlla4k-Qwen3-4B-Base-n4g8_20261019_164257.log`**:
  - `AttributeError: Can't get attribute 'BaseTaskRunner' on <module 'verl.trainer.main_ppo_v0' from '.../omni_agent/.../verl/...'>` occurred because `verl_train` had an editable `.pth` pointing to an older `verl` directory, and remote Ray actors did not inherit the shell's `PYTHONPATH`.
- **16:48 — Updated `NET_ARGS` in `scripts/run_grpo_rlla.sh`**:
  - Added `+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=${VERL_ROOT}:${PROJECT_ROOT}` so all Ray workers across all 4 nodes import `/home/ma-user/work/LB/verl` and `/home/ma-user/work/LB/github/agent-rl-lab`.

---

## [2026-10-19 16:35] — Initial Repository & Stage 1 (`ToolRL` / `RLLA-4k`) Setup

- **16:35 — Initialized Project Structure (`/home/ma-user/work/LB/github/agent-rl-lab`)**:
  - Created `scripts/download_rlla4k.sh` and downloaded `data/rlla_4k/train.parquet` (3,920 samples) and `data/rlla_4k/test.parquet` (80 samples).
  - Implemented `rewards/rlla_reward.py` (`compute_score` with `format_reward`, `accuracy_reward`, `tool_name_match`, `exact_match`, and `is_collapsed`).
  - Added `scripts/inspect_dataset.py` and `scripts/run_grpo_rlla.sh`.
