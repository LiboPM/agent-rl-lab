# Changelog

All notable changes, bug fixes, and experimental findings in `agent-rl-lab` are documented in this file with exact `YYYY-MM-DD HH:MM` (UTC+8) timestamps.

---

## [2026-11-03 19:42] — Completed 8-Epoch Pure RL Continuation (`v2` Reward) & Full 1,320-Sample Evaluation

- **11-02 16:18 — Completed 150-Step (+5 Epochs = 9 Epochs Total) Pure RL Continuation (`grpo-rlla4k-v2reward-PureRL-cont5ep-n4g8`)**:
  - Merged peak checkpoint `global_step_120` (Epoch 8.0) into `outputs/merged/grpo-rlla4k-v2reward-PureRL-ep8-step120`.
- **11-03 19:40 — Evaluated 8-Epoch Pure RL (`v2` Reward) Across All 9 In-Domain & 6 Out-of-Domain Benchmarks**:
  - **In-Domain RLLA-4k Improvements (4 Epochs $\rightarrow$ 8 Epochs v2)**:
    - Overall Reward (v1 scale): `+2.611 -> +2.676` (`Strict EM: 56.25% -> 57.50%`, `Tool Name Match: 91.7% -> 92.1%`).
    - **In-Domain 2-Tool Parallel (`N=23`)**: `Strict EM` jumped from **`47.83%` to `56.52%` (+8.69%)**, reward `+2.693 -> +2.980`.
    - **In-Domain Hammer Obfuscated APIs (`N=10`)**: `Strict EM` jumped from **`70.00%` to `80.00%` (+10.00%)**, reward `+3.717 -> +3.845` (surpassing Pure SFT's `+3.829`).
    - Successfully pruned hallucinated optional parameter keys on test samples (e.g. Sample 40 removed extra `monitoring_end_date`, Sample 79 removed extra `7F_mc3K4`, Samples 43 & 63 flipped to `Exact Match = 1.0`).
  - **Out-of-Domain BFCL-v3 Improvements (`N=1,000` Tool-Call Set)**:
    - `Strict EM` rose from **`79.70%` to `81.50%` (+1.80%)** (`Schema-Normalized EM: 83.40%`), widening the lead over Pure SFT (`60.40%`) to **+21.10%**!
    - `BFCL Simple`: `80.50% -> 82.75%` (+2.25%)
    - `BFCL Multiple`: `78.50% -> 81.50%` (+3.00%)
    - `BFCL Parallel-Multiple`: `75.00% -> 76.50%` (+1.50%, vs Pure SFT `41.50%` = **+35.00% lead**).

---

## [2026-10-31 16:14] — Implemented `rewards/rlla_reward_v2.py` & `scripts/run_grpo_rlla_v2.sh` (+5 Epochs Pure RL Continuation)

- **15:30 — Expanded Evaluation to 1,320 Samples Across 6 Benchmarks (`eval/run_multi_benchmark_eval.py`, `eval/summarize_multi_benchmark.py`)**:
  - Evaluated all models across 9 In-Domain RLLA-4k sub-categories (N=80) and 5 Out-of-Domain BFCL-v3 benchmarks (N=1,240).
  - Found that Pure RL (Step 120) outperforms Pure SFT on 1,000 OOD BFCL tool-calling questions by **+19.30%** (`79.70%` vs `60.40%`, and `75.00%` vs `41.50%` on `Parallel-Multiple`), while diagnosing 3 reward bottlenecks holding back Pure RL's In-Domain Exact Match (`56.25%`) and Irrelevance abstention (`22.50%`).
- **16:12 — Created `rewards/rlla_reward_v2.py` & `scripts/run_grpo_rlla_v2.sh`**:
  - **Fix 1 (Extra Parameter Key Penalty)**: Added `-0.75` penalty per redundant parameter key in `pd_params.keys() - gt_params.keys()` and capped non-exact tool accuracy reward at `+2.4` (vs `+3.0` for exact match), lowering the score of extra-key outputs from `+3.50` to `+2.38`.
  - **Fix 2 (Two-Tier Type-Aware Value Matching)**: Awards `+1.0` for strict `==` match and `+0.6` for primitive-normalized match (`1` vs `"1"`), eliminating false-negative gradient noise while preserving a `+0.80` total score margin for exact type match.
  - **Fix 3 (Balanced No-Tool / Abstention Reward)**: Raised valid no-tool `<response>` reward from `+1.0` (`acc=0.0`) to `+3.5` (`acc=+2.5`).

---

## [2026-10-22 17:41] — Diagnosed Full-SFT Entropy Collapse & Built DeepSeek-R1 Style Cold-Start SFT (`sft-coldstart320`)

- **17:25 — Diagnosed Why Full-Dataset SFT (`grpo-rlla4k-Qwen3-4B-SFT-ep2-n4g8`) Stalled During Subsequent GRPO**:
  - **100% Data Overlap & Boilerplate `<think>`**: All 3,920 rows in `rlla_4k/train.parquet` only contain 2 fixed 1-sentence `<think>` templates (`"I should use the appropriate tool..."` / `"I should directly respond..."`). Training 2 full epochs (`loss: 0.775 -> 0.048`) overwrote the Base model's genuine multi-step reasoning with this boilerplate and collapsed initial policy entropy (`actor/entropy: 0.72 -> 0.08`).
  - **Zero Group Variance (`std(r) = 0`)**: With `entropy = 0.08`, 75/80 validation outputs at Step 120 were 100% character-for-character identical to Step 0 (`reward: +2.9558 -> +2.9370`, `exact_match: 73.75% -> 72.50%`).
- **17:30 — Merged Zero-RL Step 120 Checkpoint (`outputs/merged/grpo-rlla4k-Qwen3-4B-Base-rlla-step120`)**:
  - Merged the 32 FSDP2 shards of `grpo-rlla4k-Qwen3-4B-Base-rlla-n4g8/global_step_120/actor` into HuggingFace `model.safetensors` via `python -m verl.model_merger merge`.
- **17:37 — Created `scripts/build_coldstart_sft_data.py` & Distilled 320 Genuine-Reasoning Cold-Start Trajectories**:
  - Ran vLLM Rejection Sampling on candidate training prompts using the merged Zero-RL Step 120 model, filtering for trajectories that achieve `format_reward == 1.0`, `exact_match == 1.0`, and genuine multi-sentence reasoning (`len(think) >= 60`, non-boilerplate).
  - Saved 320 verified cold-start samples (`285` tool-call + `35` no-tool) to `data/rlla_4k/sft_coldstart_train.json` (`rlla_4k_sft_coldstart`, ~8.1% of the training set, leaving >91% of training prompts unseen for GRPO).
- **17:41 — Trained 1-Epoch Cold-Start SFT Model (`outputs/sft/sft-coldstart320-Qwen3-4B-Base-rlla-ep1`)**:
  - Configured `configs/sft_coldstart_qwen3_4b_base_rlla.yaml` and `scripts/run_sft_coldstart_rlla.sh` (`lr=5e-6`, `num_train_epochs=1.0`, 5 optimization steps at `global_batch_size=64`).
  - Training loss decreased gently (`0.3169 -> 0.2773`, `eval_loss = 0.7020`), teaching strict XML tag closure + genuine `<think>` reasoning without collapsing policy entropy.

---

## [2026-10-20 10:26] — Added LLaMA-Factory Full-Parameter SFT Pipeline & Passed 8-GPU Smoke Test

- **10:19 — Created `scripts/prepare_sft_data.py` & `data/rlla_4k/dataset_info.json`**:
  - Converts `data/rlla_4k/train.parquet` (3,920 samples) and `test.parquet` (80 samples) into ShareGPT JSON (`sft_train.json` / `sft_test.json`) with 100% reward self-check (`format_reward == 1.0`, `exact_match == 1.0`).
- **10:21 — Created `configs/sft_qwen3_4b_base_rlla.yaml` & `scripts/run_sft_rlla.sh`**:
  - Uses LLaMA-Factory's native `parse_template()` on `models/Qwen3-4B-Base-rlla` (`template` omitted so `replace_eos=False`), preserving `eos_token=<|endoftext|>` (`151643`) and guaranteeing 100% token-level `prompt_ids` / `response_ids` alignment with VeRL `RLHFDataset`.
  - Configured single-node 8× A800-80GB DeepSpeed ZeRO-2 full-parameter SFT (`global_batch_size=64`, `lr=1e-5`, `num_train_epochs=2.0`, ~122 total steps).
- **10:26 — Passed 2-Step 8-GPU Smoke Test & Checkpoint Verification**:
  - Verified distributed training (`loss: 0.732 -> 0.7314`), `model.safetensors` saving, and `chat_template.jinja` / `tokenizer_config.json` compatibility with VeRL.

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
