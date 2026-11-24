<div align="center">

# Supervised Fine-Tuning vs. Reinforcement Learning for LLM Tool Use: A Three-Stage Empirical Study Across RLLA-4k, Expanded In-Domain 500, and BFCL-v3

[![Back to README (EN)](https://img.shields.io/badge/README-English-2563eb?style=for-the-badge)](../README.md)
[![中文版深度报告](https://img.shields.io/badge/报告-中文深度剖析-059669?style=for-the-badge)](SFT_vs_RL_Tool_Use_Deep_Dive_zh.md)

[**English Version (Current)**](SFT_vs_RL_Tool_Use_Deep_Dive.md) &nbsp;|&nbsp; [**简体中文**](SFT_vs_RL_Tool_Use_Deep_Dive_zh.md)

</div>

---

## Abstract

When adapting a pretrained base language model (`Qwen3-4B-Base`) into a reliable **Function Calling / Tool-Use Agent**, researchers and practitioners face three fundamental questions:
1. **Stage 1 — Supervised Fine-Tuning (`Pure SFT`) Overfitting & Generalization**: Should SFT checkpoints be selected via training loss (`train_loss`) or validation loss (`eval_loss`), and why does full-dataset SFT achieve strong In-Domain accuracy while degrading Out-of-Domain (OOD) generalization?
2. **Stage 2 — Direct Reinforcement Learning (`Zero-RL GRPO`) & Anti-Hacking Reward Engineering**: Why does outcome-based Group Relative Policy Optimization (`GRPO`) preserve `100%` genuine multi-step `<think>` reasoning on the exact same dataset where SFT collapses to `0.0%` genuine Chain-of-Thought (CoT)? How can dense reward shaping (`v1` $\rightarrow$ `v2`) eliminate optional-parameter reward hacking?
3. **Stage 3 — Resolving the `SFT + GRPO` Policy Entropy Collapse Paradox**: Why does initializing GRPO from a converged Full-SFT model stall completely, and how does a lightweight, genuine-reasoning **Cold-Start SFT (`320` samples, `1` Epoch) + GRPO** pipeline preserve policy entropy and achieve positive synergy?

In this study, we train and evaluate **8 controlled model checkpoints** on `Qwen3-4B-Base` using **LLaMA-Factory (SFT)** and **VeRL (GRPO)** across a 32-GPU (4 × 8 NVIDIA A800-80GB) cluster. We evaluate all models across **4 benchmark suites totaling `3,171` test instances**—including the original `RLLA-4k` test split (`80`), a newly constructed **`500`-sample AST-verified Expanded In-Domain Test Set**, `BFCL-v3 Standard` (`1,240`), and `BFCL-v3 Live` (`1,351`)—and introduce a **7-category Error Taxonomy (`E1`–`E7`)** to diagnose every failure mode quantitatively.

![Comprehensive 4-Benchmark Overview Across 2,931 Tool-Calling Samples](../assets/fig1_sft_vs_rl_overview.png)

---

## Table of Contents

1. [Introduction & Three-Stage Experimental Protocol](#1-introduction--three-stage-experimental-protocol)
2. [Benchmark Suite Design & 500-Sample Expanded In-Domain Construction (`N = 3,171`)](#2-benchmark-suite-design--500-sample-expanded-in-domain-construction-n--3171)
3. [Stage 1: Supervised Fine-Tuning (`Pure SFT`) & The Overfitting Inflection Point (`1.0 Ep` vs. `2.0 Ep`)](#3-stage-1-supervised-fine-tuning-pure-sft--the-overfitting-inflection-point-10-ep-vs-20-ep)
4. [Stage 2: Pure Reinforcement Learning (`Zero-RL GRPO`), Anti-Hacking Reward (`v1` $\rightarrow$ `v2`), & Peak Selection](#4-stage-2-pure-reinforcement-learning-zero-rl-grpo-anti-hacking-reward-v1-rightarrow-v2--peak-selection)
5. [Stage 3: Combining `SFT + GRPO` — Resolving the Policy Entropy Collapse Paradox](#5-stage-3-combining-sft--grpo--resolving-the-policy-entropy-collapse-paradox)
6. [In-Domain Label Quality Audit & Where RL Outperforms SFT Within Fixed Domains](#6-in-domain-label-quality-audit--where-rl-outperforms-sft-within-fixed-domains)
7. [Quantitative Error Taxonomy (`E1`–`E7`) & Case-by-Case Qualitative Analysis](#7-quantitative-error-taxonomy-e1e7--case-by-case-qualitative-analysis)
8. [Reward Construction Without Ground Truth (No-GT) & Production Selection Guidelines](#8-reward-construction-without-ground-truth-no-gt--production-selection-guidelines)

---

## 1. Introduction & Three-Stage Experimental Protocol

Comparing SFT and RL fairly requires eliminating two common methodological confounders:
1. **Arbitrary Checkpoint Selection**: Comparing an over-trained SFT model or an under-/over-stepped RL checkpoint distorts the true capability boundary of each paradigm.
2. **Small or Noisy Evaluation Splits**: Relying solely on the 80-sample `RLLA-4k` test split conflates genuine tool-calling capability with memorization of dataset truncation bugs.

We address both issues through a rigorous **Three-Stage Experimental Protocol**:

![Three-Stage Experimental Progression: SFT Loss, RL Reward, and Policy Entropy](../assets/fig2_training_dynamics_entropy_collapse.png)

- **Vocabulary Alignment (`Qwen3-4B-Base-rlla`)**: Native `Qwen3-4B-Base` reserves 25 special tokens (`151644`–`151668`, such as `<|im_start|>` and `<tool_call>`) whose embeddings were never updated during pretraining. Using `scripts/prepare_model_and_data.py`, we strip these untrained tokens and retain only the `151,644` pretrained BPE tokens plus `<|endoftext|>` (`151643`), ensuring all models generate XML tags (`<think>`, `<tool_call>`, `<response>`) via standard subwords.
- **Validation-Driven Checkpoint Selection**:
  - In **Stage 1 (SFT)**, we evaluate both the minimum validation loss checkpoint (`1.0 Ep / Step 62`, `eval_loss = 0.1582`) and the minimum training loss checkpoint (`2.0 Ep / Step 124`, `train_loss = 0.0196`).
  - In **Stage 2 (Pure RL)** and **Stage 3 (SFT + GRPO)**, we evaluate validation reward (`val-core/rlla/reward/mean@1`) every 10 steps (`30 steps = 1.0 Epoch`) and select the validation convergence peak (`Pure RL v1 @ Step 120`, `Pure RL v2 @ Step 120`, `Cold-Start SFT + GRPO @ Step 100`).

---

## 2. Benchmark Suite Design & 500-Sample Expanded In-Domain Construction (`N = 3,171`)

All models are trained exclusively on the `RLLA-4k` training split (`3,920` samples covering `4,109` unique tool schemas) and evaluated across **4 benchmark suites (`3,171` samples)**:

| Benchmark Suite | Sample Size | Domain Relationship | Structural Composition & Evaluation Focus |
| :--- | :---: | :---: | :--- |
| **1. `RLLA-4k` Original Test Set** | `80` | **In-Domain** (Original Split) | `36` Single-Tool + `35` Multi-Tool + `9` No-Tool Abstention (`64` Clean AST + `16` Noisy/Truncated GT). |
| **2. `RLLA-4k` Expanded In-Domain Test Set** (`test_indomain_500.parquet`) | **`500`** (`6.25×` larger) | **Strictly In-Domain** (100% Seen Tools, 0% Query Overlap) | `250` Single-Tool + `250` Multi-Tool (`250` xLAM + `250` ToolACE). **100% of candidate tools appear in `train.parquet`**, **100% pass AST validation**. |
| **3. `BFCL-v3` Standard Benchmark** | `1,240` (`1,000` Call + `240` Irrel) | **Out-of-Domain** (Unseen Schemas) | `400` Simple + `200` Multiple + `200` Parallel + `200` Parallel-Multiple + `240` Irrelevance Abstention. |
| **4. `BFCL-v3 Live` Real-World Benchmark** | `1,351` | **Out-of-Domain** (Real Enterprise Queries) | `258` Live-Simple + `1,053` Live-Multiple + `16` Live-Parallel + `24` Live-Parallel-Multiple (`10–20` candidate tools per prompt). |

### Master Evaluation Table Across All 8 Models (`Strict EM%` / `Schema-Normalized EM%`)

| Model ID & Training Paradigm | Compute & Wall Time | RLLA-4k Original (`N=80`) | Expanded In-Domain (`N=500`) | BFCL-v3 Standard (`N=1,000`) | BFCL-v3 Live (`N=1,351`) | BFCL Irrelevance (`N=240`) | Genuine `<think>` CoT Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`0. Zero-Shot Base`** (`Qwen3-4B-Base-rlla`) | 0 steps | `43.75%` / `50.00%` | `52.20%` / `53.00%` | `69.50%` / `72.60%` | `57.81%` / `63.73%` | `19.17%` | `99.7%` (`290` chars) |
| **`1a. Pure SFT (1.0 Ep, Min Eval-Loss=0.1582)`** | 8 GPUs, 5m | **`73.75%` / `75.00%`** | `64.60%` / `69.00%` | `60.20%` / `74.80%` | `52.70%` / `66.17%` | `76.25%` | `0.0%` (`87` chars, Fake) |
| **`1b. Pure SFT (2.0 Ep, Min Train-Loss=0.0196)`** | 8 GPUs, 10m | **`73.75%` / `75.00%`** | **`65.60%` / `69.80%`** | `60.40%` / `74.10%` | `49.30%` / `62.92%` | **`89.17%`** | `0.0%` (`87` chars, Fake) |
| **`2a. Full-SFT (2.0 Ep) + GRPO (Step 120, Neg. Ctrl)`** | 10m + 38m | `72.50%` / `75.00%` | `65.20%` / `69.60%` | `59.50%` / `73.60%` | `49.30%` / `63.06%` | `88.33%` | `0.0%` (`87` chars, Fake) |
| **`2b. Cold-Start SFT (320 Genuine-CoT, 1.0 Ep)`** | 8 GPUs, 1m | `48.75%` / `55.00%` | `60.00%` / `60.40%` | `79.30%` / `81.60%` | `58.92%` / `64.99%` | `16.25%` | `99.8%` (`249` chars) |
| **`2c. Cold-Start SFT (320) + GRPO (Step 100, Pos. Ctrl)`** | 1m + 32m | `50.00%` / `57.50%` | `62.60%` / `63.00%` | `81.20%` / `83.00%` | **`58.99%`** / `65.21%` | `15.83%` | **`100.0%`** (`268` chars) |
| **`3a. Pure RL (Zero-RL GRPO, 4.0 Ep, v1 Reward)`** | 32 GPUs, 38m | `53.75%` / `60.00%` | `59.00%` / `59.60%` | `79.70%` / `82.10%` | `58.33%` / `64.91%` | **`22.50%`** | **`100.0%`** (`275` chars) |
| **`3b. Pure RL (Zero-RL GRPO, 8.0 Ep, v2 Reward)`** | 32 GPUs, 75m | `56.25%` / `61.25%` | `63.20%` / `63.20%` | **`81.50%` / `83.40%`** | `58.77%` / **`65.58%`** | `20.83%` | **`100.0%`** (`281` chars) |

---

## 3. Stage 1: Supervised Fine-Tuning (`Pure SFT`) & The Overfitting Inflection Point (`1.0 Ep` vs. `2.0 Ep`)

### 3.1 Token-Level Cross-Entropy Objective & Loss Trajectory
Standard SFT minimizes the token-level negative log-likelihood over the complete target string $y = (y_1, \dots, y_T)$ (including both `<think>` and `<tool_call>`):

$$\mathcal{L}_{\text{SFT}}(\theta) = -\frac{1}{T}\sum_{t=1}^{T} \log \pi_\theta(y_t \mid x, y_{<t})$$

Tracking `outputs/sft/sft-rlla4k-Qwen3-4B-Base-rlla-ep2/trainer_log.jsonl` across 2.0 Epochs (`124` steps, batch size `64`) reveals a clear overfitting inflection point at **Step 62 (`1.0 Epoch`)**:

| SFT Step (Epoch) | Training Loss (`loss`) | Validation Loss (`eval_loss`) | Learning Rate | Gradient Norm | Checkpoint Status |
| :---: | :---: | :---: | :---: | :---: | :--- |
| `Step 10` (`0.16 Ep`) | `0.7852` | — | `1.97e-5` | `1.05` | Learning XML tags (`<think>`, `<tool_call>`) |
| `Step 30` (`0.48 Ep`) | `0.2143` | — | `1.63e-5` | `0.41` | Memorizing static `<think>` template |
| `Step 50` (`0.81 Ep`) | `0.1559` | — | `1.30e-5` | `0.33` | Converging on tool call syntax |
| **`Step 62` (`1.00 Ep`)** | `0.1510` | **`0.1582` (Global Min)** | `1.05e-5` | `0.29` | **`Model 1a` (`checkpoint-62`, Best OOD SFT)** |
| `Step 90` (`1.45 Ep`) | `0.0369` | — | `3.50e-6` | `0.26` | Overfitting dataset-specific string quoting |
| **`Step 124` (`2.00 Ep`)** | **`0.0196`** | `0.1700` (`+7.5%` Rebound) | `0.00` | `0.22` | **`Model 1b` (`checkpoint-124`, Over-trained SFT)** |

### 3.2 Why Over-Training SFT Hurts OOD Generalization & Destroys CoT
1. **The Cost of Epoch 2 Overfitting (`-3.40%` on `BFCL-Live`)**: While training from `1.0 Ep` to `2.0 Ep` yields a `+1.00%` gain on `Expanded In-Domain 500` (`64.60%` $\rightarrow$ `65.60%`), it causes a **`-3.40%` drop on `BFCL-Live` (`52.70%` $\rightarrow$ `49.30%`)**. Our Error Taxonomy (Section 7) proves that Epoch 2 memorizes closed-world refusal boundaries so aggressively that it triggers **`49` False Abstention errors (`E2`)** on unfamiliar `BFCL-Live` tools (up `9.8×` from `5` cases at Epoch 1).
2. **Destruction of Pretrained Reasoning (`0.0%` Genuine CoT)**: Because `3,895 / 3,920` (`99.36%`) of `RLLA-4k` training targets contain the identical 16-word placeholder `<think> I should use the appropriate tool with proper parameters to respond to the user's need. </think>`, token-level teacher forcing forces both `Model 1a` and `Model 1b` to emit this exact boilerplate string on `100%` of test prompts.

---

## 4. Stage 2: Pure Reinforcement Learning (`Zero-RL GRPO`), Anti-Hacking Reward (`v1` $\rightarrow$ `v2`), & Peak Selection

### 4.1 Outcome-Based Verification Preserves `100%` Genuine Chain-of-Thought
In **Zero-RL GRPO**, we optimize `Qwen3-4B-Base-rlla` directly using group-relative advantages over $G=8$ sampled rollouts $\{o_1, \dots, o_G\}$:

$$A_i = \frac{r_i - \text{mean}(\{r_1, \dots, r_G\})}{\text{std}(\{r_1, \dots, r_G\}) + \epsilon}$$

Crucially, our rule-based verifier (`rewards/rlla_reward_v2.py`) **never compares the model's `<think>` content against the boilerplate `<think>` string in Ground Truth**. It checks `<think>...</think>` only for structural tag closure (`+1.0` format reward) and evaluates accuracy (`[-3.0, +3.0]`) purely on the parsed `<tool_call>` JSON AST. Rollouts that perform genuine arithmetic derivation and schema checking inside `<think>` produce more accurate JSON arguments, earning higher rewards and positive advantages ($A_i > 0$).

### 4.2 Six-Stage Anti-Reward-Hacking Design (`rlla_reward_v2.py`)

![Reward Engineering & Quantitative Error Taxonomy](../assets/fig4_reward_v2_staircase.png)

Error taxonomy diagnosis on `Model 3a (Pure RL, 4.0 Ep, v1 Reward)` revealed `50` cases of **`E5_Hallucinated_Extra_Param_Key`** on `Expanded In-Domain 500`—the policy learned to speculatively populate optional schema parameters because `v1` deducted only `-0.25/extra_key` and allowed non-exact match scores up to `+3.75`.

In `rewards/rlla_reward_v2.py`, we define the per-tool accuracy score $s_k$ with two anti-reward-hacking mechanisms:

$$s_k = \underbrace{0.25}_{\text{Tool Name}} + \underbrace{0.20 \times \frac{|K_{\text{pred}} \cap K_{\text{gt}}|}{|K_{\text{gt}}|}}_{\text{Required Key Recall}} + \underbrace{0.55 \times \frac{\sum_{p \in K_{\text{gt}}} \mathbb{I}(\text{val\_match}(p))}{|K_{\text{gt}}|}}_{\text{Value Match}} - \underbrace{0.25 \times |K_{\text{pred}} \setminus K_{\text{gt}}|}_{\text{Anti-Hacking 1: Extra-Key Penalty}}$$

1. **Anti-Hacking Mechanism 1 (`-0.75 / extra_key` Penalty)**: Scaled to $[-3.0, +3.0]$, each redundant optional key incurs a **`-0.75` penalty**, making speculative optional-key filling strictly negative-expected-value.
2. **Anti-Hacking Mechanism 2 (`+3.40` Non-Exact Cap vs. `+4.00` Exact Match)**: Any prediction with `exact_match < 1.0` is hard-capped at **`+2.40` accuracy (`+3.40` total)**, preserving a **`+0.60` step-function bonus** exclusively for 100% strict exact matches (`+4.00`).

Selecting the `Step 120` validation peak (`val_reward = +2.7421`, `Model 3b`) cuts `E5` extra-key hallucinations by **`-28%` (`50` $\rightarrow$ `36`)**, improves `Expanded In-Domain 500` Strict EM by **`+4.20%` (`59.00%` $\rightarrow$ `63.20%`)**, and achieves **`81.50%` Strict EM (`83.40%` Norm EM)** on `BFCL-v3 Standard`.

---

## 5. Stage 3: Combining `SFT + GRPO` — Resolving the Policy Entropy Collapse Paradox

### 5.1 Negative Control (`Model 2a: Full-SFT 2Ep + GRPO`) — Why Standard `SFT -> RL` Stalls
When initializing GRPO from `Model 1b (Full-SFT 2Ep)`, token-level policy entropy starts at **`0.085`** (collapsed). Across $G=8$ rollouts at $T=1.0$, all 8 trajectories emit the identical boilerplate `<think>` and identical `<tool_call>` JSON:

$$o_1 = o_2 = \dots = o_8 \implies \text{std}(\{r_1, \dots, r_8\}) = 0 \implies A_i \equiv 0$$

With zero intra-group reward variance, policy gradients vanish (`val_reward`: `+2.9558` $\rightarrow$ `+2.9370`, with `75 / 80` validation outputs at `Step 120` identical to `Step 0`).

### 5.2 Positive Control (`Model 2c: Cold-Start SFT 320 + GRPO v2`) — Restoring Exploration & Synergy
To prevent entropy collapse while retaining SFT's format warm-up benefit, we train `Qwen3-4B-Base-rlla` for **1.0 Epoch (`5` gradient steps, `lr = 1e-5`)** on **`320` rejection-sampled trajectories containing genuine multi-step `<think>` reasoning** (`Model 2b`), and then run 120 steps of **VeRL GRPO (`v2` reward)** (`logs/metrics/grpo-rlla4k-v2reward-ColdStartSFT320-n4g8_20261121_104610.jsonl`):

| GRPO Step (`Epoch`) | Validation Reward (`val_reward`) | Format Compliance (`fmt`) | Tool Name Match (`tool_name`) | Validation Strict EM | Policy Entropy (`actor/entropy`) | Progression Notes |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Step 0` (`0.00 Ep`)** | `+2.5112` | `93.75%` | `88.75%` | `50.00%` | **`0.492` (Healthy)** | 5-step Cold-Start SFT eliminates format collapse (`+0.4090` Base) |
| `Step 10` (`0.33 Ep`) | `+2.5898` | `93.75%` | `90.83%` | `51.25%` | `0.421` | Rapid tool selection improvement |
| `Step 30` (`1.00 Ep`) | `+2.6504` | `93.75%` | `90.83%` | `51.25%` | `0.318` | Matches 4-Epoch Pure RL in just 1 Epoch (30 steps) |
| `Step 40` (`1.33 Ep`) | `+2.6941` | `95.00%` | `90.83%` | `52.50%` | `0.284` | Format compliance reaches `95.0%` |
| `Step 90` (`3.00 Ep`) | `+2.7079` | `95.00%` | `90.83%` | **`53.75%`** | `0.228` | Multi-tool accuracy climbs steadily |
| **`Step 100` (`3.33 Ep`)** | **`+2.7486` (Global Peak!)** | **`96.25%` (Peak)** | **`91.46%` (Peak)** | `52.50%` | `0.215` | **Selected as `Model 2c` (Peak Val Reward / Format / Tool Name)** |
| `Step 120` (`4.00 Ep`) | `+2.7136` | `95.00%` | `91.04%` | `50.00%` | `0.202` | Slight post-peak dip; early-stopped at Step 100 |

Evaluating `Model 2c (Step 100)` across all 4 benchmark suites confirms substantial **`SFT + GRPO` synergy over `Cold-Start SFT (Model 2b)` alone**:
- **`Expanded In-Domain 500`**: Strict EM improves from `60.00%` to **`62.60%` (`+2.60%`)**, with **`Multi-Tool (250)` jumping by `+6.40%` (`48.40%` $\rightarrow$ `54.80%`)**;
- **`BFCL-v3 Standard (1,000)`**: Strict EM improves from `79.30%` to **`81.20%` (`83.00%` Norm EM)**, with **`Parallel-Multiple (200)` jumping by `+5.50%` (`70.00%` $\rightarrow$ `75.50%`)**;
- **`BFCL-v3 Live (1,351)`**: Achieves **`58.99%` Strict EM (`797 / 1,351`)**, the **highest score among all 8 models**.

---

## 6. In-Domain Label Quality Audit & Where RL Outperforms SFT Within Fixed Domains

![In-Domain Label Audit & Out-of-Domain Complexity Scaling](../assets/fig3_deep_dive_clean_dirty_and_bfcl.png)

### 6.1 Sub-Benchmark Breakdown Across In-Domain & Out-of-Domain Suites

| Benchmark Subset | Count | `0. Base` | `1a. SFT (1Ep)` | `1b. SFT (2Ep)` | `2a. Full-SFT+RL` | `2b. Cold-SFT` | `2c. Cold-SFT+RL` | `3a. Pure RL v1` | `3b. Pure RL v2` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`RLLA-80` Clean Valid GT** | `64` | `54.69%` | `79.69%` | **`79.69%`** | `78.12%` | `60.94%` | `62.50%` | `67.19%` | **`71.88%`** |
| **`RLLA-80` Noisy / Truncated GT** | `16` | `0.00%` | `50.00%` | **`50.00%`** | `50.00%` | `0.00%` | `0.00%` | `0.00%` | **`0.00%`** |
| **`InDom-500` Single-Tool** | `250` | `55.60%` | `69.60%` | **`71.60%`** | `71.60%` | **`71.60%`** | `70.40%` | `68.40%` | `70.80%` |
| **`InDom-500` Multi-Tool ($\ge 2$)** | `250` | `48.80%` | **`59.60%`** | **`59.60%`** | `58.80%` | `48.40%` | `54.80%` | `49.60%` | **`55.60%`** |
| **`InDom-500` xLAM (`Clean Types`)** | `250` | `69.60%` | `82.80%` | **`83.60%`** | `83.20%` | `80.80%` | **`82.40%`** | `79.20%` | **`82.80%`** |
| **`InDom-500` ToolACE (`Quoted Ints`)** | `250` | `34.80%` | `46.40%` | **`47.60%`** | `47.20%` | `39.20%` | `42.80%` | `38.80%` | **`43.60%`** |
| **`BFCL-v3` Simple** | `400` | `81.75%` | `68.75%` | `67.50%` | `66.25%` | `81.00%` | `82.00%` | `80.50%` | **`82.75%`** |
| **`BFCL-v3` Multiple** | `200` | `77.50%` | `59.50%` | `61.00%` | `61.00%` | **`82.50%`** | `82.00%` | `78.50%` | `81.50%` |
| **`BFCL-v3` Parallel** | `200` | `74.00%` | `64.00%` | `64.50%` | `63.50%` | `82.00%` | **`84.50%`** | `84.00%` | `84.00%` |
| **`BFCL-v3` Parallel-Multiple** | `200` | `32.50%` | `40.00%` | `41.50%` | `40.50%` | `70.00%` | `75.50%` | `75.00%` | **`76.50%`** |

1. **Why SFT Appears to Lead on Raw `RLLA-80`**: `16 / 80` (`20%`) of `RLLA-80` Ground Truths contain truncated unclosed JSON strings (e.g., `"[{"subject": "Math"`) or string-quoted integers. SFT memorizes these dataset bugs (`50.0%` on broken GTs), whereas RL always outputs valid schema-typed JSON (`0.0%` on broken GTs).
2. **Near-Parity on Clean In-Domain `xLAM-250`**: On the `250` unseen in-domain queries from `xLAM` where Ground Truth types strictly follow the JSON Schema, **`Pure RL (v2)` (`82.80%`) and `Cold-Start SFT + GRPO` (`82.40%`) nearly match `Pure SFT` (`83.60%`)** without full-dataset SFT!

---

## 7. Quantitative Error Taxonomy (`E1`–`E7`) & Case-by-Case Qualitative Analysis

### 7.1 Error Taxonomy Tables (`eval/error_taxonomy_report.json`)

#### Table 7-A: Out-of-Domain `BFCL-v3 Standard` (`N = 1,000` Tool-Calling Queries)

| Model | `CORRECT` (Strict EM) | `E1` Format/JSON | `E2` False Abstain | `E3a` Wrong Tool | `E3b` Call Count | `E4` Missing Key | `E5` Extra Key | `E6` Value Error | `E7` Type/Quote Only |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`0. Zero-Shot Base`** | `695` (`69.5%`) | `8` | `2` | `8` | **`150`** | `4` | `0` | `102` | `31` |
| **`1a. Pure SFT (1.0 Ep)`** | `602` (`60.2%`) | `6` | `0` | `2` | `32` | **`69`** | `2` | `141` | **`146`** |
| **`1b. Pure SFT (2.0 Ep)`** | `604` (`60.4%`) | `14` | `2` | `3` | `28` | **`52`** | `2` | `158` | **`137`** |
| **`2a. Full-SFT + GRPO`** | `595` (`59.5%`) | `14` | `2` | `3` | `25` | **`55`** | `2` | `164` | **`140`** |
| **`2b. Cold-Start SFT (320)`** | `793` (`79.3%`) | `7` | `1` | `7` | `45` | `6` | `0` | `118` | `23` |
| **`2c. Cold-Start SFT + GRPO`** | **`812` (`81.2%`)** | **`4`** | **`0`** | **`6`** | `27` (**`-40%`**) | **`6`** | `1` | `126` | **`18`** |
| **`3a. Pure RL (4.0 Ep, v1)`** | `797` (`79.7%`) | `5` | `2` | `8` | `28` | `6` | `0` | `130` | `24` |
| **`3b. Pure RL (8.0 Ep, v2)`** | **`815` (`81.5%`)** | `5` | `2` | `7` | **`18`** (**`-36%`**) | `8` | `1` | **`125`** | `19` |

#### Table 7-B: Out-of-Domain `BFCL-v3 Live` (`N = 1,351` Real-World Multi-Candidate Queries)

| Model | `CORRECT` (Strict EM) | `E1` Format/JSON | `E2` False Abstain | `E3a` Wrong Tool | `E3b` Call Count | `E4` Missing Key | `E5` Extra Key | `E6` Value Error | `E7` Type/Quote Only |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`0. Zero-Shot Base`** | `781` (`57.8%`) | `3` | `3` | `57` | `21` | `32` | `17` | `357` | `80` |
| **`1a. Pure SFT (1.0 Ep)`** | `712` (`52.7%`) | `14` | `5` | `27` | `22` | **`86`** | `10` | `293` | **`182`** |
| **`1b. Pure SFT (2.0 Ep)`** | `666` (`49.3%`) | `11` | **`49` (`9.8×` surge)** | `25` | `21` | `66` | `11` | `318` | **`184`** |
| **`2c. Cold-Start SFT + GRPO`** | **`797` (`59.0%`)** | **`1`** | **`0`** | **`47`** | `27` | `48` | **`13`** | `334` | `84` |
| **`3b. Pure RL (8.0 Ep, v2)`** | `794` (`58.8%`) | **`1`** | `3` | `50` | `29` | `49` | `15` | **`318`** | `92` |

#### Table 7-C: Strictly In-Domain `Expanded In-Domain 500` (`N = 500` Unseen Queries)

| Model | `CORRECT` (Strict EM) | `E1` Format/JSON | `E2` Mode Error | `E3a` Wrong Tool | `E3b` Call Count | `E4` Missing Key | `E5` Extra Key | `E6` Value Error | `E7` Type/Quote Only |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`0. Zero-Shot Base`** | `261` (`52.2%`) | **`235` (No XML)** | `0` | `0` | `0` | `0` | `0` | `0` | `4` |
| **`1b. Pure SFT (2.0 Ep)`** | **`328` (`65.6%`)** | `3` | `5` | `1` | `15` | `35` | `14` | **`78`** | `21` |
| **`2b. Cold-Start SFT (320)`** | `300` (`60.0%`) | `15` | `1` | `3` | `29` | `7` | `49` | `93` | `2` |
| **`2c. Cold-Start SFT + GRPO`** | `313` (`62.6%`) | `15` | `1` | `3` | `21` (**`-28%`**) | `9` | `39` (**`-20%`**) | `97` | `2` |
| **`3a. Pure RL (4.0 Ep, v1)`** | `295` (`59.0%`) | `16` | `1` | `6` | `21` | `8` | `50` | `100` | `3` |
| **`3b. Pure RL (8.0 Ep, v2)`** | `316` (`63.2%`) | `16` | `0` | `4` | `19` | `11` | **`36` (`-28%`)** | `98` | **`0`** |

### 7.2 Qualitative Case Studies
1. **In-Domain Arithmetic Offset (`rlla_test_20`)**: Prompt asks for 10 templates starting from the **20th template** (`skip` is 0-indexed). `Pure SFT` emits boilerplate `<think>` and copies `"skip": 20` (`E6`, score `0.0`). `Pure RL` and `Cold-Start SFT + GRPO` compute `20 - 1 = 19` inside `<think>` and emit `"skip": 19` (score `1.0`).
2. **In-Domain Inclusive Range (`rlla_test_55`)**: Prompt asks for simulation from **Year 70 to Year 80 inclusive**. `Pure SFT` outputs `"years": 10` (`E6`, score `0.0`), whereas `Pure RL` and `Cold-Start SFT + GRPO` derive `80 - 70 + 1 = 11` inside `<think>` and output `"years": 11` (score `1.0`).
3. **In-Domain 3-Tool Completeness (`rlla_test_34`)**: `Pure SFT` drops all parameters on the 3rd parallel tool call (`"parameters": {}`, `E4`), whereas `Pure RL` and `Cold-Start SFT + GRPO` populate all 3 tool calls completely (score `1.0`).

---

## 8. Reward Construction Without Ground Truth (No-GT) & Production Selection Guidelines

### 8.1 Three-Tier Verification Pipeline When No Ground Truth Exists in Production
When scaling RL to unannotated production logs where human Ground Truth `<tool_call>` labels are unavailable, `rlla_reward_v2.py` generalizes into a **3-Tier Verifier Pipeline**:
1. **Tier 1 — Static JSON Schema & AST Verifier (Zero Annotation Cost)**: Validates XML format, tool name existence, presence of all `required` keys, and strict parameter data types (`integer`/`number`/`boolean`/`enum`) against the tool's JSON Schema. Eliminates `E1`, `E4`, and `E7` with zero human labels.
2. **Tier 2 — Sandbox Dry-Run / Execution Outcome Verifier**: Executes read-only API calls or mock dry-runs and rewards HTTP 200 / non-empty return payloads.
3. **Tier 3 — Group-Relative LLM-as-a-Judge**: Invokes a judge model only on schema-valid rollouts within a GRPO group that disagree on argument values (`E5` / `E6`).

### 8.2 Summary Decision Matrix

| Deployment Scenario | Recommended Paradigm | Empirical Justification |
| :--- | :--- | :--- |
| **Closed, Static Internal APIs (<20 tools, strict latency cap)** | **`Pure SFT (1.0 Ep, Min Eval-Loss)`** | Reaches `79.69%` on clean `RLLA-80` in `5–10 mins` on 8 GPUs with short outputs (`~328` chars); early-stopping at `1.0 Ep` prevents Epoch 2 `E2` false abstention surge. |
| **Open-World / MCP Tool Ecosystems (100+ unseen third-party schemas)** | **`Pure RL (Zero-RL GRPO, v2 Reward)`** | Achieves **`81.50%` Strict EM (`83.40%` Norm EM)** on `BFCL-v3 Standard` (`+21.10%` over SFT) and **`82.80%`** on clean In-Domain `xLAM-250` with `100%` genuine CoT. |
| **Enterprise Multi-Candidate Assistants (Fast RL Convergence + Highest Live Robustness)** | **`Cold-Start SFT (320 Genuine-CoT, 1 Ep) + GRPO (v2)`** | Reaches `93.75%` format compliance at `Step 0` while preserving entropy (`0.492`), peaks at `Step 100` (`+2.7486`), and achieves the **highest `BFCL-Live` accuracy (`58.99%` Strict EM)** across all 8 models. |
