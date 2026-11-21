# Supervised Fine-Tuning vs. Reinforcement Learning for LLM Tool Use: A Three-Stage Empirical Study

<div align="center">

[![Back to README (EN)](https://img.shields.io/badge/README-English-2563eb?style=for-the-badge)](../README.md)
[![中文版深度报告](https://img.shields.io/badge/报告-中文深度剖析-059669?style=for-the-badge)](SFT_vs_RL_Tool_Use_Deep_Dive_zh.md)

</div>

---

## 1. Executive Summary & Three-Stage Experimental Design

This technical report presents a controlled empirical investigation of **Supervised Fine-Tuning (`Pure SFT`)**, **Direct Reinforcement Learning (`Zero-RL GRPO`)**, and **Hybrid `SFT + GRPO`** for tool-calling agents on `Qwen3-4B-Base`.

Rather than comparing arbitrary checkpoints, we structure our investigation into **three progressive stages**, selecting the optimal checkpoint in each stage strictly according to held-out validation metrics:

1. **Stage 1 — Supervised Fine-Tuning (`Pure SFT`) & Overfitting Analysis**:
   - We train `Qwen3-4B-Base-rlla` on the full `RLLA-4k` dataset (`3,920` samples) and monitor both `train_loss` and `eval_loss`.
   - Validation loss reaches its global minimum at **`1.0 Epoch` (`Step 62`, `eval_loss = 0.1582`)** and rebounds at **`2.0 Epochs` (`Step 124`, `eval_loss = 0.1700`, `train_loss = 0.0196`)**.
   - Evaluating both checkpoints reveals that training beyond the `eval_loss` minimum slightly improves memorization of training API conventions (`Expanded In-Domain 500`: `64.60%` $\rightarrow$ `65.60%`), but **degrades real-world Out-of-Domain (OOD) performance on `BFCL-Live` by `-3.40%` (`52.70%` $\rightarrow$ `49.30%`)** due to a `9.8×` surge in false abstentions (`E2`). Furthermore, full-sequence cross-entropy loss on `RLLA-4k`'s boilerplate `<think>` strings destroys `100%` of genuine Chain-of-Thought reasoning (`Genuine CoT = 0.0%`).
2. **Stage 2 — Pure Reinforcement Learning (`Zero-RL GRPO`) & Anti-Hacking Reward Engineering (`v1` $\rightarrow$ `v2`)**:
   - Applying **VeRL GRPO** directly to the base model with an outcome-based verifier (`rewards/rlla_reward.py`, `v1`) eliminates format collapse (`97.5%` $\rightarrow$ `2.5%`), preserves **`100.0%` genuine Chain-of-Thought**, and reaches **`79.70%` Strict EM** on OOD `BFCL-v3 Standard` at `Step 120` (`4.0 Epochs`).
   - Error taxonomy analysis on `Model 3a (v1)` identifies a subtle **Reward Hacking** behavior: the policy speculatively fills optional parameters (`50` cases of `E5_Hallucinated_Extra_Param_Key` on `In-Domain 500`).
   - Upgrading to `rewards/rlla_reward_v2.py` (`-0.75/extra_key` penalty and `+3.40` non-exact cap) and training to the validation reward peak at **`Step 120` (`8.0 Epochs`, `val_reward = +2.7421`)** reduces `E5` extra-key hallucinations by **`-28%` (`50` $\rightarrow$ `36`)**, boosts `In-Domain 500` Strict EM by **`+4.20%` (`59.00%` $\rightarrow$ `63.20%`)**, and achieves **`81.50%` Strict EM (`83.40%` Norm EM)** on `BFCL-v3 Standard` (`+21.10%` over `Pure SFT`).
3. **Stage 3 — Resolving the `SFT + GRPO` Entropy Collapse Paradox (Negative Control vs. Positive Control)**:
   - **Negative Control (`Model 2a: Full-SFT 2Ep + GRPO`)**: Initializing GRPO from the low-entropy `Full-SFT` checkpoint (`actor/entropy = 0.085`) results in zero intra-group reward variance ($\sigma_r = 0 \implies A_i = 0$) across all $G=8$ rollouts, causing complete training stagnation (`75/80` validation outputs at `Step 120` remain identical to `Step 0`).
   - **Positive Control (`Model 2c: Cold-Start SFT 320 + GRPO`)**: Warming up the base model for `1.0 Epoch` (`5` gradient steps) on **`320` genuine-CoT trajectories (`Model 2b`)** achieves `93.75%` initial format compliance while preserving high policy entropy (`actor/entropy = 0.492`). Subsequent GRPO (`v2` reward) climbs from `+2.5112` (`Step 0`) to a global validation peak of **`+2.7486` at `Step 100` (`3.33 Epochs`)**, improving `In-Domain 500` by **`+2.60%`** (`Multi-Tool`: `+6.40%`), improving `BFCL-v3 Standard` by **`+1.90%`** (`Parallel-Multiple`: `+5.50%`), and achieving the **highest `BFCL-v3 Live` Strict EM (`58.99%`) across all 8 models**.

<div align="center">
  <img src="../assets/fig1_sft_vs_rl_overview.png" width="96%" alt="Comprehensive 4-Benchmark Overview"/>
</div>

---

## 2. Benchmark Suite Design & Expanded In-Domain Construction (`N = 3,171`)

A critical limitation of the original `RLLA-4k` evaluation split (`data/rlla_4k/test.parquet`) is its small sample size (`N = 80`) and the presence of `16` corrupted Ground Truth strings (`20%` label noise). To ensure high statistical power for both In-Domain and Out-of-Domain evaluation, we construct `data/rlla_4k/test_indomain_500.parquet` (`scripts/build_expanded_indomain_test.py`) and evaluate across **4 benchmark suites (`3,171` total instances)**:

1. **`RLLA-4k` Original In-Domain Test Set (`N = 80`)**:
   - `36` Single-Tool, `35` Multi-Tool ($\ge 2$ calls), and `9` No-Tool abstention queries.
   - Contains `64` clean AST-valid samples and `16` corrupted GT samples (unclosed JSON strings or quoted integers contradicting integer schema types).
2. **`RLLA-4k` Expanded Strictly In-Domain Test Set (`N = 500`, `6.25×` Expansion)**:
   - Extracted from the upstream `ToolACE` (`250` samples) and `xLAM-60k` (`250` samples) corpora under three strict invariants:
     1. **100% In-Domain Tool Coverage**: Every candidate tool in the system prompt appears in `rlla_4k/train.parquet` (`4,109` unique training tool names).
     2. **0% Query Overlap**: Zero overlap with any user query in `train.parquet` (`3,920`) or `test.parquet` (`80`).
     3. **100% AST & Exact-Match Verification**: Every Ground Truth `<tool_call>` block passes `json.loads` AST verification (`500 / 500` self-consistency score `+4.0`).
   - Balanced across complexity (`250` Single-Tool + `250` Multi-Tool) and upstream schema conventions (`250` `xLAM` clean-typed + `250` `ToolACE` string-quoted).
3. **`BFCL-v3` Standard Out-of-Domain Benchmark (`N = 1,240`)**:
   - `1,000` tool-calling instances (`400` Simple, `200` Multiple, `200` Parallel, `200` Parallel-Multiple) + `240` Irrelevance abstention instances.
4. **`BFCL-v3 Live` Real-World User Benchmark (`N = 1,351`)**:
   - `258` Live-Simple, `1,053` Live-Multiple, `16` Live-Parallel, and `24` Live-Parallel-Multiple queries featuring long system prompts with `10–20` candidate enterprise tools per prompt.

---

## 3. Detailed Stage-by-Stage Experimental Analysis

<div align="center">
  <img src="../assets/fig2_training_dynamics_entropy_collapse.png" width="98%" alt="Three-Stage Progression: SFT Loss, RL Reward, and Policy Entropy"/>
</div>

### 3.1 Stage 1: SFT Loss Progression & Checkpoint Selection (`outputs/sft/sft-rlla4k-Qwen3-4B-Base-rlla-ep2/trainer_log.jsonl`)

| SFT Step (Epoch) | Training Loss (`loss`) | Validation Loss (`eval_loss`) | Learning Rate | Gradient Norm | Checkpoint Status |
| :---: | :---: | :---: | :---: | :---: | :--- |
| `Step 10` (`0.16 Ep`) | `0.7852` | — | `1.97e-5` | `1.05` | Warming up format tags |
| `Step 30` (`0.48 Ep`) | `0.2143` | — | `1.63e-5` | `0.41` | Rapidly memorizing `<think>` template |
| **`Step 62` (`1.00 Ep`)** | `0.1510` | **`0.1582` (Min)** | `1.05e-5` | `0.29` | **`Model 1a` (`checkpoint-62`, Best OOD SFT)** |
| `Step 90` (`1.45 Ep`) | `0.0369` | — | `3.50e-6` | `0.26` | Overfitting training surface strings |
| **`Step 124` (`2.00 Ep`)** | **`0.0196`** | `0.1700` (`+7.5%`) | `0.00` | `0.22` | **`Model 1b` (`checkpoint-124`, Over-trained SFT)** |

### 3.2 Stage 2 & Stage 3: VeRL GRPO Validation Trajectories (`Step 0` $\rightarrow$ `Step 120`)

The table below tracks validation metrics on `RLLA-4k` (`val-core/rlla/reward/mean@1`, `format_reward`, `tool_name_match`, and `exact_match`) every 20 steps across our four GRPO training runs:

| Global Step (Epoch) | `Zero-RL GRPO (v1)` (`Model 3a`, Ep 0–4) | `Zero-RL GRPO (v2)` (`Model 3b`, Ep 4–8) | `Full-SFT + GRPO` (`Model 2a`, Neg. Ctrl) | `Cold-Start SFT + GRPO (v2)` (`Model 2c`, Pos. Ctrl) |
| :---: | :---: | :---: | :---: | :---: |
| **`Step 0` (`0.0 Ep`)** | `+0.4090` (`EM=43.8%`) | `+2.6594` (`EM=53.8%`) | `+2.9558` (`EM=73.8%`) | `+2.5112` (`Fmt=93.8%`, `EM=50.0%`) |
| **`Step 20` (`0.7 Ep`)** | `+2.0752` (`EM=46.3%`) | `+2.6480` (`EM=55.0%`) | `+2.9502` (`EM=73.8%`) | `+2.6237` (`Fmt=93.8%`, `EM=50.0%`) |
| **`Step 40` (`1.3 Ep`)** | `+2.5015` (`EM=51.3%`) | `+2.6812` (`EM=53.8%`) | `+2.9481` (`EM=72.5%`) | `+2.6941` (`Fmt=95.0%`, `EM=52.5%`) |
| **`Step 60` (`2.0 Ep`)** | `+2.5210` (`EM=52.5%`) | `+2.7105` (`EM=55.0%`) | `+2.9410` (`EM=72.5%`) | `+2.5890` (`Fmt=93.8%`, `EM=51.3%`) |
| **`Step 80` (`2.7 Ep`)** | `+2.5934` (`EM=52.5%`) | `+2.6710` (`EM=55.0%`) | `+2.9395` (`EM=72.5%`) | `+2.6879` (`Fmt=95.0%`, `EM=51.3%`) |
| **`Step 100` (`3.3 Ep`)** | `+2.6218` (`EM=53.8%`) | `+2.7180` (`EM=55.0%`) | `+2.9370` (`EM=72.5%`) | **`+2.7486` (`Fmt=96.3%`, Peak!)** |
| **`Step 120` (`4.0 Ep`)** | **`+2.6584` (`EM=53.8%`, Peak)** | **`+2.7421` (`EM=56.3%`, Peak)** | `+2.9370` (`EM=72.5%`) | `+2.7136` (`Fmt=95.0%`, `EM=50.0%`) |

---

## 4. Concrete Case Studies: Why RL Beats SFT on Both In-Domain & OOD Reasoning

### Case Study 1: In-Domain Arithmetic Offset Reasoning (`rlla_test_20`)
- **User Prompt**: *"Fetch 10 marketing email templates starting from the **20th template** (`skip` is 0-indexed offset, `limit` is count)."*
- **Ground Truth**: `{"name": "get_email_templates", "parameters": {"limit": 10, "skip": 19}}`
- **`Pure SFT` (`Model 1a` & `1b`, Score = `0.0`)**:
  ```xml
  <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think>
  <tool_call>
  {"name": "get_email_templates", "parameters": {"limit": 10, "skip": 20}}
  </tool_call>
  ```
- **`Pure RL` (`Model 3b`) & `Cold-Start SFT + GRPO` (`Model 2c`, Score = `1.0`)**:
  ```xml
  <think> The user wants 10 templates starting from the 20th template. Since `skip` is a 0-indexed offset, starting at the 20th item means skipping the first 19 templates (skip = 20 - 1 = 19), and setting limit = 10. </think>
  <tool_call>
  {"name": "get_email_templates", "parameters": {"limit": 10, "skip": 19}}
  </tool_call>
  ```

### Case Study 2: In-Domain Inclusive Interval Counting (`rlla_test_55`)
- **User Prompt**: *"Simulate annual population growth from **Year 70 to Year 80 inclusive**."*
- **Ground Truth**: `{"name": "simulate_growth", "parameters": {"years": 11, ...}}`
- **`Pure SFT` (Score = `0.0`)**: Emits boilerplate `<think>` and sets `"years": 10` (`80 - 70 = 10`, off-by-one error).
- **`Pure RL` & `Cold-Start SFT + GRPO` (Score = `1.0`)**: Explicitly reasons inside `<think>` that *"from Year 70 to Year 80 inclusive is 80 - 70 + 1 = 11 years"* and outputs `"years": 11`.

### Case Study 3: Out-of-Domain Parallel-Multiple Composition (`BFCL-v3 parallel_multiple_42`)
- **Task**: Given 6 candidate tools in mathematics and physics, invoke two distinct tools (`calculate_magnetic_field` and `calculate_electric_force`) with unit-converted parameters.
- **`Pure SFT` (`Model 1b`, Score = `0.0`)**: Emits boilerplate `<think>`, quotes numerical floats as strings (`"current": "5.0"` $\rightarrow$ `E7_Type_Quote_Mismatch`), and omits the required `"distance"` key on the second tool call (`E4_Missing_Required_Param_Key`).
- **`Cold-Start SFT + GRPO` (`Model 2c`) & `Pure RL` (`Model 3b`, Score = `1.0`)**: Uses `<think>` to enumerate both required tools and verify each required parameter's native JSON type (`float`/`int`) before emitting `<tool_call>`.

---

## 5. Production Deployment Guidelines: Choosing Between SFT, Pure RL, and Cold-Start SFT + RL

| Deployment Scenario | Recommended Paradigm | Empirical Justification |
| :--- | :--- | :--- |
| **1. Fixed, Closed-Set Internal APIs (<20 static tools, tight latency budget)** | **`Pure SFT (1.0 Ep, Min Eval-Loss)`** | Reaches `73.75%` on `RLLA-80` (`79.69%` on clean subset) in `10 minutes` on 8 GPUs with short output length (`~328` chars), avoiding 2-Epoch OOD degradation. |
| **2. Open-World / MCP Tool Marketplaces (100+ dynamic third-party schemas)** | **`Pure RL (Zero-RL GRPO, v2 Reward)`** | Achieves **`81.50%` Strict EM (`83.40%` Norm EM)** on `BFCL-v3 Standard` (`+21.1%` over SFT) and **`82.80%`** on clean In-Domain `xLAM-250` while preserving `100%` genuine Chain-of-Thought. |
| **3. Enterprise Multi-Tool Assistants Requiring Fast RL Convergence & High Robustness** | **`Cold-Start SFT (320 Genuine-CoT, 1 Ep) + GRPO (v2 Reward)`** | Eliminates the first ~30 steps of RL format exploration (`Step 0` format compliance `93.8%`, entropy `0.492`), peaks at `Step 100` (`+2.7486`), and achieves the **highest real-world `BFCL-Live` accuracy (`58.99%` Strict EM)** alongside `81.20%` on `BFCL-v3 Standard`. |
