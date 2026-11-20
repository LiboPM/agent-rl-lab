# Agent-RL-Lab: Tool-Use SFT vs. RL (VeRL GRPO + LLaMA-Factory)

**English** | [简体中文](README_zh.md)

An empirical study and reproducible training pipeline comparing **Supervised Fine-Tuning (LLaMA-Factory)** and **Reinforcement Learning with Verifiable Rewards (VeRL / GRPO)** for LLM Function Calling and Tool Use.

- **Full Technical Report**: [English (`docs/SFT_vs_RL_Tool_Use_Deep_Dive.md`)](docs/SFT_vs_RL_Tool_Use_Deep_Dive.md) | [简体中文 (`docs/SFT_vs_RL_Tool_Use_Deep_Dive_zh.md`)](docs/SFT_vs_RL_Tool_Use_Deep_Dive_zh.md)

---

## 1. Overview & Benchmark Results

Starting from **Qwen3-4B-Base**, we train five controlled model variants on the exact same tool-calling dataset (`ToolRL / RLLA-4k`, 3,920 training samples) and evaluate them across **three benchmarks (2,431 test samples total)**:

1. **RLLA-4k Test (`N=80`, In-Domain)**: Held-out split from the same source distribution as the training set (comprising 64 clean samples and 16 samples with upstream dataset conversion artifacts).
2. **BFCL-v3 Standard (`N=1,000`, Out-of-Domain)**: UC Berkeley Function Calling Leaderboard benchmark (`Simple 400`, `Multiple 200`, `Parallel 200`, `Parallel-Multiple 200`) featuring unseen APIs.
3. **BFCL-v3 Live (`N=1,351`, Out-of-Domain Real-World)**: Real-world user queries containing 10–20 candidate tool schemas per prompt.

![In-Domain Memorization (SFT) vs. Out-of-Domain Generalization (RL)](assets/fig1_sft_vs_rl_overview.png)

### Summary of Controlled Experiments

| Model Variant | Compute & Wall Time | `RLLA-4k` In-Domain<br>(All 80 / Clean 64) | `BFCL-v3` Standard OOD<br>(N=1,000 · Strict / Norm EM) | `BFCL-Live` Real-World OOD<br>(N=1,351 · Strict / Norm EM) | Genuine CoT Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **0. Zero-Shot Base** (`Qwen3-4B-Base-rlla`) | — | `43.75%` / `54.69%` | `60.20%` / `63.60%` | `45.30%` / `51.89%` | `93.8%` |
| **1. Pure SFT** (`3,920 samples × 2 Epochs`) | 8 GPUs · 10 mins | **`73.75%`** / **`79.69%`** | `60.40%` / `64.60%` | `49.30%` / `58.92%` | `0.0%` |
| **2. Full-SFT + RL** (`2Ep SFT + 120-step GRPO`) | 10m + 1.5 hrs | `72.50%` / `78.12%` | `59.70%` / `63.90%` | `48.93%` / `58.77%` | `0.0%` |
| **3. Cold-Start SFT** (`320 samples × 1 Epoch`) | 8 GPUs · 1 min | `50.00%` / `60.94%` | `67.10%` / `70.50%` | `51.81%` / `59.44%` | `95.0%` |
| **4. Pure RL** (`Zero-RL GRPO 4 Epochs, v1`) | 32 GPUs · 1.5 hrs | `56.25%` / `70.31%` | `79.70%` / `81.90%` | `58.11%` / `64.99%` | `97.5%` |
| **5. Pure RL** (`GRPO 8 Epochs, v2 Reward`) | 32 GPUs · 3.0 hrs | `57.50%` / **`71.88%`** | **`81.50%`** / **`83.40%`** | **`58.77%`** / **`65.58%`** | **`98.8%`** |

---

## 2. Key Technical Findings

### 2.1 In-Domain Memorization vs. Out-of-Domain Generalization
- **In-Domain (`RLLA-4k`)**: When in-distribution supervision is available, **Pure SFT reaches `73.75%` in 10 minutes** (`79.69%` on the 64 clean samples), capturing dataset-specific default parameter conventions and refusal phrasing at 1/18th the compute cost of RL.
- **Out-of-Domain (`BFCL-v3` & `BFCL-Live`)**: On unseen tool definitions, Pure SFT stagnates at `60.40%` on `BFCL-v3` (+0.2% over the untrained Base model `60.20%`). **Pure RL reaches `81.50%` (+21.1% over Pure SFT)** and outperforms Pure SFT by **+35.0%** (`76.50%` vs. `41.50%`) on the hardest `Parallel-Multiple` subset.
- **Label Quality Audit on `RLLA-4k`**: AST inspection of `RLLA-4k` reveals that 16/80 test samples (20%) contain corrupted Ground Truth labels caused by naive `.split(",")` parsing during upstream dataset construction (6 samples with truncated unclosed JSON strings such as `"[{\"subject\": \"Mathematics\""`, and 10 samples with string-quoted integers/booleans violating the prompt's JSON Schema). Pure SFT achieves `50.00%` on these 16 corrupted samples by memorizing the 536 analogous truncation artifacts in the training split, whereas Pure RL emits valid JSON and scores `0.00%` against the broken labels. On the 64 clean samples, Pure RL achieves `71.88%`.

![In-Domain Label Quality Audit & Out-of-Domain Complexity Scaling](assets/fig3_deep_dive_clean_dirty_and_bfcl.png)

### 2.2 Why Full-SFT + GRPO Stalls: Boilerplate CoT and Policy Entropy Collapse
- Every sample in the raw `RLLA-4k` training set contains a static 14-word boilerplate template inside `<think>` (`"I should use the appropriate tool with proper parameters to respond to the user's need."`).
- Training 2 epochs of full-parameter SFT via token-level teacher forcing overwrites the Base model's native reasoning distribution with this template, collapsing token-level policy entropy (`actor/entropy`) from `0.72` to `0.08`.
- During subsequent GRPO training, all $G=8$ rollouts per prompt produce identical token sequences, yielding zero within-group reward variance ($\text{std}(r) = 0$) and zero policy gradient updates (75/80 validation outputs at Step 120 remain character-for-character identical to Step 0).
- In contrast, **Pure RL** evaluates only the `<tool_call>` JSON payload and leaves `<think>` unconstrained, preserving multi-step reasoning (`98.8%` genuine CoT). For SFT warm-up prior to RL, rejection-sampling 320 trajectories with genuine CoT (`Model 3: Cold-Start SFT`, 1 epoch / 5 steps) achieves `0.95` format compliance while retaining `95.0%` genuine CoT.

![Training Dynamics: Pure RL vs. Full-SFT + GRPO](assets/fig2_training_dynamics_entropy_collapse.png)

### 2.3 Dense Reward Shaping & Anti-Reward-Hacking (`rewards/rlla_reward_v2.py`)
To provide dense intermediate gradients while preventing reward hacking, `rlla_reward_v2.py` shapes rewards across `[-3.0, +4.0]` with two key mechanisms:
- **Extra-Key Penalty (`-0.75 / key`)**: Penalizes redundant optional parameter keys not present in the Ground Truth (`set(pd_params.keys()) - set(gt_params.keys())`), eliminating the strategy of guessing all optional schema fields.
- **Non-Exact Match Cap (`+3.4` vs. `+4.0`)**: Caps non-exact tool calls at `+3.4` total score, preserving a `+0.60` advantage margin exclusively for 100% strict exact match (`gt_tools == pd_tools`).

![Dense Reward Shaping & Anti-Reward-Hacking Design](assets/fig4_reward_v2_staircase.png)

### 2.4 Qwen3-Base Untrained Special Token Removal & Cross-Framework Alignment
- **Stripping Untrained Added Tokens (`scripts/prepare_qwen3_base_tokenizer.py`)**: In `Qwen3-4B-Base`, token IDs `151644..151668` (`<|im_end|>`, `<think>`, `<tool_call>`, etc.) have untrained embedding rows identical to 2,227 unused vocabulary entries (`norm=0.3574`). Removing `151644..151668` from `added_tokens` decomposes XML tags into pre-trained BPE subwords and sets `<|endoftext|>` (`151643`, `norm=1.2266`) as the sole EOS token, improving Step 0 zero-shot exact match from `0.00%` to `43.75%`.
- **Token-Exact Alignment Between LLaMA-Factory and VeRL (`configs/sft_qwen3_4b_base_rlla.yaml`)**: Omitting `template` in LLaMA-Factory forces `parse_template()` to load `tokenizer.chat_template` directly with `replace_eos=False`, guaranteeing identical `prompt_ids` and `eos_token_id` across SFT and VeRL.

---

## 3. Repository Structure

```text
agent-rl-lab/
├── assets/                                  # Publication-grade figures (300 DPI PNG)
├── docs/
│   ├── SFT_vs_RL_Tool_Use_Deep_Dive.md      # Full Technical Report (English)
│   └── SFT_vs_RL_Tool_Use_Deep_Dive_zh.md   # Full Technical Report (Chinese)
├── configs/
│   ├── sft_qwen3_4b_base_rlla.yaml          # LLaMA-Factory Full SFT config
│   └── sft_coldstart_qwen3_4b_base_rlla.yaml# 320-sample Rejection-Sampled Cold-Start SFT config
├── rewards/
│   ├── rlla_reward.py                       # Baseline ToolRL rule-based reward (v1)
│   └── rlla_reward_v2.py                    # Upgraded reward with anti-reward-hacking (v2)
├── scripts/
│   ├── download_rlla4k.sh                   # Download RLLA-4k dataset
│   ├── prepare_qwen3_base_tokenizer.py      # Strip untrained special tokens & align EOS
│   ├── prepare_sft_data.py                  # Convert Parquet to ShareGPT JSON with reward check
│   ├── build_coldstart_sft_data.py          # Distill 320 genuine-CoT cold-start trajectories
│   ├── run_sft_rlla.sh                      # Launch Full SFT (8x GPU DeepSpeed ZeRO-2)
│   ├── run_sft_coldstart_rlla.sh            # Launch Cold-Start SFT
│   ├── run_grpo_rlla.sh                     # Launch VeRL GRPO training (v1)
│   ├── run_grpo_rlla_v2.sh                  # Launch VeRL GRPO continuation (v2 reward)
│   └── generate_github_figures.py           # Reproduce all figures in assets/
└── eval/
    ├── compare_models.py                    # In-domain RLLA-4k evaluation
    ├── run_multi_benchmark_eval.py          # Joint evaluation on RLLA-4k + BFCL-v3 (1,320 samples)
    ├── run_bfcl_live_eval.py                # Evaluation on BFCL-v3 Live (1,351 samples)
    ├── summarize_multi_benchmark.py         # Aggregate JSON & Markdown evaluation reports
    ├── multi_benchmark_report.md            # Detailed benchmark breakdown table
    └── rlla4k_model_comparison.json         # Per-category RLLA-4k metrics
```

---

## 4. Quick Start

### 4.1 Environment & Tokenizer Preparation

```bash
# 1. Download RLLA-4k dataset
bash scripts/download_rlla4k.sh

# 2. Build patched Qwen3-4B-Base-rlla tokenizer
python3 scripts/prepare_qwen3_base_tokenizer.py

# 3. Convert SFT data and run 100% reward verification
python3 scripts/prepare_sft_data.py
```

### 4.2 Train with VeRL GRPO (Pure RL)

```bash
# Stage 1: Zero-RL GRPO from Base (4 Epochs)
bash scripts/run_grpo_rlla.sh

# Stage 2: Continuation with anti-reward-hacking v2 reward
bash scripts/run_grpo_rlla_v2.sh
```

### 4.3 Train with LLaMA-Factory (Full SFT / Cold-Start SFT)

```bash
# Option A: Full 2-Epoch SFT (8 GPUs, ~10 mins)
bash scripts/run_sft_rlla.sh

# Option B: 320-Sample Genuine-CoT Cold-Start SFT (8 GPUs, ~1 min)
python3 scripts/build_coldstart_sft_data.py
bash scripts/run_sft_coldstart_rlla.sh
```

### 4.4 Evaluate Across All Three Benchmarks (2,431 Samples)

```bash
# Evaluate on RLLA-4k (80) + BFCL-v3 Standard (1,240)
python3 eval/run_multi_benchmark_eval.py
python3 eval/summarize_multi_benchmark.py

# Evaluate on BFCL-v3 Live (1,351)
python3 eval/run_bfcl_live_eval.py
```
