<div align="center">

# Agent-RL-Lab: Supervised Fine-Tuning vs. Reinforcement Learning (GRPO) for LLM Tool Use

**A Controlled Comparative Study of Training Mechanisms, In-Domain vs. Out-of-Domain Boundaries, Error Taxonomies, and Trade-Offs on `Qwen3-4B-Base` (`N = 3,171`)**

[![English](https://img.shields.io/badge/Language-English-2563eb?style=for-the-badge)](README.md)
[![简体中文](https://img.shields.io/badge/语言-简体中文-059669?style=for-the-badge)](README_zh.md)
[![Deep Dive Report (EN)](https://img.shields.io/badge/Report-Technical_Deep_Dive_(EN)-7c3aed?style=for-the-badge)](docs/SFT_vs_RL_Tool_Use_Deep_Dive.md)
[![深度报告 (中文)](https://img.shields.io/badge/报告-技术深度剖析_(中文)-d97706?style=for-the-badge)](docs/SFT_vs_RL_Tool_Use_Deep_Dive_zh.md)

[![Framework: VeRL](https://img.shields.io/badge/RL_Engine-VeRL_(FSDP_+_vLLM)-10b981?style=flat-square)](https://github.com/volcengine/verl)
[![SFT: LLaMA-Factory](https://img.shields.io/badge/SFT_Engine-LLaMA--Factory_(DeepSpeed_ZeRO--2)-3b82f6?style=flat-square)](https://github.com/hiyouga/LLaMA-Factory)
[![Base Model: Qwen3-4B-Base](https://img.shields.io/badge/Base_Model-Qwen3--4B--Base-8b5cf6?style=flat-square)](https://huggingface.co/Qwen)
[![Evaluation: 3,171 Samples](https://img.shields.io/badge/Benchmarks-RLLA--80_%7C_InDomain--500_%7C_BFCL--v3_%7C_BFCL--Live-f59e0b?style=flat-square)](#2-in-domain-vs-out-of-domain-benchmark-matrix-n--3171)

</div>

---

## Core Research Question

When post-training a pretrained base model (`Qwen3-4B-Base`) into a **Tool-Calling Agent**, **how do Supervised Fine-Tuning (`SFT`) and Reinforcement Learning (`RL / GRPO`) fundamentally differ, and what are the real advantages and concrete disadvantages of RL compared to SFT?**

To ensure a fair, apples-to-apples comparison free from arbitrary checkpoint selection or single-benchmark bias, our investigation is organized around **four comparative dimensions**:
1. **Fair Comparison Prerequisite — Selecting Each Paradigm's Optimal Checkpoint**: Tracking SFT's `train_loss` vs. `eval_loss` overfitting inflection point (`1.0 Ep` vs. `2.0 Ep`), GRPO's validation `val_reward` peak (`v1` $\rightarrow$ `v2` anti-hacking reward), and the policy entropy dynamics when combining `SFT + GRPO` (`Full-SFT + GRPO` vs. `Cold-Start SFT + GRPO`).
2. **In-Domain vs. Out-of-Domain Capability Boundaries (`N = 3,171`)**: Contrasting SFT and RL across **seen In-Domain tools** (`RLLA-80` + a `500`-sample AST-verified Expanded In-Domain set), **unseen OOD tools** (`BFCL-v3 Standard 1,000` + `BFCL-Live 1,351`), and **no-tool abstention** (`BFCL Irrelevance 240`).
3. **Microscopic Failure Attribution (`E1`–`E7` Error Taxonomy)**: Decomposing exact failure modes across format syntax, false abstention, multi-tool count, missing required keys, hallucinated extra keys, arithmetic errors, and type/quote contamination.
4. **Objective Trade-Off Synthesis (Pros & Cons)**: Weighing RL's strengths (OOD generalization, reasoning retention, noise immunity) against its real costs (compute overhead, inference latency, over-triggering on irrelevant tools, and reward-hacking sensitivity).

<div align="center">
  <img src="assets/fig1_sft_vs_rl_overview.png" width="96%" alt="Comprehensive 4-Benchmark Overview Across 2,931 Tool-Calling Samples"/>
</div>

---

## 1. Fair Comparison Prerequisite: Convergence Dynamics & Optimal Checkpoint Selection for SFT, Pure RL, and SFT+RL

Comparing SFT and RL requires evaluating each paradigm at its held-out validation optimum rather than arbitrary training steps. Figure 2 illustrates the convergence and entropy dynamics across all three training paths:

<div align="center">
  <img src="assets/fig2_training_dynamics_entropy_collapse.png" width="98%" alt="Three-Stage Experimental Progression: SFT Loss, RL Reward, and Policy Entropy"/>
</div>

### Stage 1: SFT Exploration & The Overfitting Inflection Point (`Model 1a` vs. `Model 1b`)
We train `Qwen3-4B-Base-rlla` on the full `RLLA-4k` training split (`3,920` samples, global batch size `64`, `62` steps/epoch) using LLaMA-Factory and evaluate every epoch on held-out validation loss (`eval_loss`):

| SFT Checkpoint | Global Step | Training Loss (`train_loss`) | Validation Loss (`eval_loss`) | RLLA-80 Strict EM | In-Domain 500 Strict EM | BFCL-v3 (1k) Strict / Norm EM | BFCL-Live (1,351) Strict / Norm EM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Model 1a`: Pure SFT (`1.0 Ep`, Min Eval-Loss)** | `Step 62` | `0.1510` | **`0.1582` (Global Min)** | `73.75%` | `64.60%` | `60.20%` / **`74.80%`** | **`52.70%` / `66.17%`** |
| **`Model 1b`: Pure SFT (`2.0 Ep`, Min Train-Loss)** | `Step 124` | **`0.0196`** | `0.1700` (`+7.5%` Rebound) | `73.75%` | **`65.60%`** | `60.40%` / `74.10%` | `49.30%` / `62.92%` (`-3.40%`) |

- **Why `eval_loss` diverges from `train_loss` after Epoch 1.0**: Between Step 62 (`1.0 Ep`) and Step 124 (`2.0 Ep`), training loss drops by `87%` (`0.1510` $\rightarrow$ `0.0196`), but validation loss rebounds from **`0.1582` to `0.1700`**.
- **In-Domain Memorization vs. OOD Degradation**: Training for a second epoch slightly improves accuracy on unseen queries sharing the exact same training tools (`Expanded In-Domain 500`: `64.60%` $\rightarrow$ `65.60%`), but **degrades real-world OOD accuracy on `BFCL-Live` by `-3.40%` (`52.70%` $\rightarrow$ `49.30%`)**, triggered by a **`9.8×` surge in False Abstention errors (`E2`: `5` $\rightarrow$ `49` cases)**.
- **Loss of Chain-of-Thought (`0.0%` Genuine CoT)**: Because `99.4%` (`3,895 / 3,920`) of `RLLA-4k` training targets contain a static 16-word boilerplate string inside `<think>` (`"I should use the appropriate tool with proper parameters to respond to the user's need."`), sequence-level cross-entropy loss forces both `1.0 Ep` and `2.0 Ep` SFT models to overwrite pretrained reasoning with this template (`100%` boilerplate output).

---

### Stage 2: Pure RL (`Zero-RL GRPO`) Exploration, Reward Engineering (`v1` $\rightarrow$ `v2`), & Peak Selection (`Model 3a` vs. `Model 3b`)
Instead of supervised imitation, we apply **VeRL GRPO** (`G = 8` rollouts per prompt, `KL = 0.001`) directly to `Qwen3-4B-Base-rlla`, evaluating validation reward (`val-core/rlla/reward/mean@1`) every 10 steps (`30 steps = 1.0 Epoch`):

| RL Stage & Reward Function | Training Horizon | Val Reward Trajectory (`Step 0` $\rightarrow$ Peak $\rightarrow$ End) | Selected Peak Checkpoint | In-Domain 500 Strict EM | BFCL-v3 (1k) Strict / Norm EM | BFCL-Live Strict / Norm EM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Model 3a`: Pure RL (`v1` Reward)** | `0 -> 4.0 Ep` (`120` steps) | `+0.4090` $\rightarrow$ **`+2.6584` (`Step 120`)** | `Step 120` (`4.0 Ep`) | `59.00%` | `79.70%` / `82.10%` | `58.33%` / `64.91%` |
| **`Model 3b`: Pure RL (`v2` Anti-Hacking)** | `4.0 -> 9.0 Ep` (`150` steps) | `+2.6594` $\rightarrow$ **`+2.7421` (`Step 120`)** $\rightarrow$ `+2.7077` (`Step 150`) | **`Step 120` (`8.0 Ep`)** | **`63.20%` (`+4.20%`)** | **`81.50%` / `83.40%`** | **`58.77%` / `65.58%`** |

- **Why Pure RL Preserves `100%` Genuine CoT**: Our outcome-based verifier (`rewards/rlla_reward_v2.py`) parses `<think>...</think>` only to verify structural presence (`+1.0` format reward) and evaluates accuracy (`[-3.0, +3.0]`) strictly on the `<tool_call>` JSON AST. It **never computes token loss against the boilerplate `<think>` text in Ground Truth**. Rollouts containing genuine step-by-step arithmetic and schema checks achieve higher parameter accuracy and receive positive group-relative advantages ($A_i > 0$).
- **Diagnosing & Fixing Reward Hacking (`v1` $\rightarrow$ `v2`)**: Error taxonomy analysis on `Model 3a (v1 Reward)` revealed `50` cases of **`E5_Hallucinated_Extra_Param_Key`** on `In-Domain 500` (speculatively filling optional parameters with default values because `v1` deducted only `-0.25/extra_key` and allowed non-exact match scores up to `+3.75`). In `rewards/rlla_reward_v2.py`, we increased the extra-key penalty to **`-0.75/key`** and capped non-exact accuracy at **`+2.40` (total `+3.40` vs. `+4.00` for exact match)**. Continuing training to the `Step 120` validation peak (`+2.7421`) cuts `E5` extra-key hallucinations by **`-28%` (`50` $\rightarrow$ `36`)** and boosts `In-Domain 500` Strict EM by **`+4.20%` (`59.00%` $\rightarrow$ `63.20%`)**.

---

### Stage 3: Combining `SFT + GRPO` — Resolving the Entropy Collapse Paradox (`Model 2a` Negative Control vs. `Model 2c` Positive Control)
Why do standard `SFT -> RL` pipelines often stall in tool-calling tasks, and how should SFT be designed to genuinely benefit subsequent RL? We run a controlled comparison on the exact same 32-GPU VeRL GRPO setup:

| Pipeline Design | Initial Policy Entropy (`Step 0`) | GRPO Validation Reward (`Step 0` $\rightarrow$ Peak) | Genuine CoT Rate | In-Domain 500 Strict EM | BFCL-v3 (1k) Strict / Norm EM | BFCL-Live Strict / Norm EM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Model 2a`: Full-SFT (`2Ep`) + GRPO (Negative Control)** | `0.085` (Collapsed) | `+2.9558` $\rightarrow$ `+2.9370` (Stalled) | `0.0%` | `65.20%` (`-0.40%` vs SFT) | `59.50%` / `73.60%` (`-0.90%`) | `49.30%` / `63.06%` (`+0.00%`) |
| **`Model 2b`: Cold-Start SFT (`320` Genuine-CoT, `1Ep`)** | `0.492` (Healthy) | `+2.5112` (at `Step 0`) | `99.8%` | `60.00%` | `79.30%` / `81.60%` | `58.92%` / `64.99%` |
| **`Model 2c`: Cold-Start SFT (`320`) + GRPO (`Step 100`, Positive Control)** | `0.492` $\rightarrow$ `0.215` | `+2.5112` $\rightarrow$ **`+2.7486` (`Step 100` Peak)** | **`100.0%`** | **`62.60%` (`+2.60%` gain)** | **`81.20%` / `83.00%` (`+1.90%` gain)** | **`58.99%` / `65.21%` (Best Overall)** |

1. **Why `Full-SFT (2Ep) + GRPO` Fails (`Model 2a`)**:
   - After 2 epochs of SFT on boilerplate `<think>`, token-level policy entropy collapses to `0.08–0.12`.
   - When GRPO samples $G=8$ responses per prompt at `temperature=1.0`, almost all 8 trajectories emit the exact same boilerplate `<think>` and identical `<tool_call>` JSON:
     $$\sigma_r = \sqrt{\frac{1}{G}\sum_{i=1}^{G}(r_i - \mu_r)^2} = 0 \implies A_i = \frac{r_i - \mu_r}{\sigma_r + \epsilon} = 0$$
   - With zero intra-group variance, policy gradients vanish. At `Step 120`, `75 / 80` validation predictions are character-for-character identical to `Step 0`.
2. **How `Cold-Start SFT (320) + GRPO` Succeeds (`Model 2c`)**:
   - We curate **`320` high-quality trajectories (`data/sft_coldstart_320/train_coldstart_320.json`)** containing **genuine multi-step `<think>` reasoning** and train for **only `1.0 Epoch` (`5` gradient steps, `lr = 1e-5`)**.
   - This lightweight warm-up immediately raises initial XML format compliance from `2.5%` (Base) to **`93.75%`** (`Step 0` validation reward **`+2.5112`** vs. `+0.4090` for Base) while preserving high policy entropy (**`0.492`**).
   - Subsequent GRPO (`v2` reward) actively explores diverse tool compositions, climbing steadily from **`+2.5112` (`Step 0`) $\rightarrow$ `+2.6504` (`Step 30`) $\rightarrow$ `+2.7079` (`Step 90`) $\rightarrow$ `+2.7486` (`Step 100` Global Peak)**, improving `In-Domain 500` by **`+2.60%`** (`Multi-Tool`: `48.40%` $\rightarrow$ **`54.80%`**, `+6.40%`), improving `BFCL-v3 Standard` by **`+1.90%`** (`Parallel-Multiple`: `70.00%` $\rightarrow$ **`75.50%`**, `+5.50%`), and achieving the **highest `BFCL-Live` Strict EM (`58.99%`) across all 8 models**!

---

## 2. In-Domain vs. Out-of-Domain Benchmark Matrix (`N = 3,171`)

To compare SFT and RL across **seen In-Domain APIs**, **unseen Out-of-Domain schemas**, and **no-tool abstention** without small-sample bias, we evaluate all 8 models across **4 complementary benchmark suites (`3,171` samples)**:

| Benchmark Suite | Sample Size | Domain Relationship | Structural Composition & Evaluation Focus |
| :--- | :---: | :---: | :--- |
| **1. `RLLA-4k` Original Test Set** | `80` | **In-Domain** (Original Split) | `36` Single-Tool + `35` Multi-Tool + `9` No-Tool Abstention (`64` Clean AST + `16` Noisy/Truncated GT). |
| **2. `RLLA-4k` Expanded In-Domain Test Set** (`test_indomain_500.parquet`) | **`500`** (`6.25×` larger) | **Strictly In-Domain** (100% Seen Tools, 0% Query Overlap) | `250` Single-Tool + `250` Multi-Tool (`250` xLAM + `250` ToolACE). **100% of candidate tools appear in `train.parquet`**, **100% pass AST validation**. |
| **3. `BFCL-v3` Standard Benchmark** | `1,240` (`1,000` Call + `240` Irrel) | **Out-of-Domain** (Unseen Schemas) | `400` Simple + `200` Multiple + `200` Parallel + `200` Parallel-Multiple + `240` Irrelevance Abstention. |
| **4. `BFCL-v3 Live` Real-World Benchmark** | `1,351` | **Out-of-Domain** (Real Enterprise Queries) | `258` Live-Simple + `1,053` Live-Multiple + `16` Live-Parallel + `24` Live-Parallel-Multiple (`10–20` candidate tools per prompt). |

### Master Evaluation Table Across All 8 Models (`Strict EM%` / `Schema-Normalized EM%`)

| Model ID & Training Paradigm | RLLA-4k Original (`N=80`) | Expanded In-Domain (`N=500`) | BFCL-v3 Standard (`N=1,000`) | BFCL-v3 Live (`N=1,351`) | BFCL Irrelevance (`N=240`) | Genuine `<think>` CoT Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`0. Zero-Shot Base`** (`Qwen3-4B-Base-rlla`) | `43.75%` / `50.00%` | `52.20%` / `53.00%` | `69.50%` / `72.60%` | `57.81%` / `63.73%` | `19.17%` | `99.7%` (`290` chars) |
| **`1a. Pure SFT (1.0 Ep, Min Eval-Loss=0.1582)`** | **`73.75%` / `75.00%`** | `64.60%` / `69.00%` | `60.20%` / `74.80%` | `52.70%` / `66.17%` | `76.25%` | `0.0%` (`87` chars, Fake) |
| **`1b. Pure SFT (2.0 Ep, Min Train-Loss=0.0196)`** | **`73.75%` / `75.00%`** | **`65.60%` / `69.80%`** | `60.40%` / `74.10%` | `49.30%` / `62.92%` | **`89.17%`** | `0.0%` (`87` chars, Fake) |
| **`2a. Full-SFT (2.0 Ep) + GRPO (Step 120, Neg. Ctrl)`** | `72.50%` / `75.00%` | `65.20%` / `69.60%` | `59.50%` / `73.60%` | `49.30%` / `63.06%` | `88.33%` | `0.0%` (`87` chars, Fake) |
| **`2b. Cold-Start SFT (320 Genuine-CoT, 1.0 Ep)`** | `48.75%` / `55.00%` | `60.00%` / `60.40%` | `79.30%` / `81.60%` | `58.92%` / `64.99%` | `16.25%` | `99.8%` (`249` chars) |
| **`2c. Cold-Start SFT (320) + GRPO (Step 100, Pos. Ctrl)`** | `50.00%` / `57.50%` | `62.60%` / `63.00%` | `81.20%` / `83.00%` | **`58.99%`** / `65.21%` | `15.83%` | **`100.0%`** (`268` chars) |
| **`3a. Pure RL (Zero-RL GRPO, 4.0 Ep, v1 Reward)`** | `53.75%` / `60.00%` | `59.00%` / `59.60%` | `79.70%` / `82.10%` | `58.33%` / `64.91%` | **`22.50%`** | **`100.0%`** (`275` chars) |
| **`3b. Pure RL (Zero-RL GRPO, 8.0 Ep, v2 Reward)`** | `56.25%` / `61.25%` | `63.20%` / `63.20%` | **`81.50%` / `83.40%`** | `58.77%` / **`65.58%`** | `20.83%` | **`100.0%`** (`281` chars) |

---

## 3. Deconstructing In-Domain Performance & OOD Complexity Scaling

Why does `Pure SFT` appear to lead `Pure RL` by `+17.5%` on `RLLA-4k (80)` (`73.75%` vs. `56.25%`), yet only by `+2.4%` on `Expanded In-Domain (500)` (`65.60%` vs. `63.20%`), while trailing by **`-21.1%` on `BFCL-v3 Standard (1,000)`** (`60.40%` vs. `81.50%`)? Figure 3 reveals the exact structural mechanisms:

<div align="center">
  <img src="assets/fig3_deep_dive_clean_dirty_and_bfcl.png" width="98%" alt="In-Domain Label Audit & Out-of-Domain Complexity Scaling"/>
</div>

### 3.1 In-Domain Sub-Category Breakdown (`Expanded In-Domain N=500` & `RLLA-80`)

| In-Domain Subset | Sample Count | `0. Base` | `1a. SFT (1Ep)` | `1b. SFT (2Ep)` | `2b. Cold-Start SFT` | `2c. Cold-Start SFT + GRPO` | `3a. Pure RL (v1)` | `3b. Pure RL (v2)` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`RLLA-80` Clean Valid GT** | `64` | `54.69%` | `79.69%` | **`79.69%`** | `60.94%` | `62.50%` | `67.19%` | **`71.88%`** |
| **`RLLA-80` Noisy / Truncated GT** | `16` | `0.00%` | `50.00%` | **`50.00%`** | `0.00%` | `0.00%` | `0.00%` | **`0.00%`** |
| **`InDom-500` Single-Tool** | `250` | `55.60%` | `69.60%` | `71.60%` | **`71.60%`** | `70.40%` | `68.40%` | `70.80%` |
| **`InDom-500` Multi-Tool ($\ge 2$)** | `250` | `48.80%` | **`59.60%`** | **`59.60%`** | `48.40%` | `54.80%` (`+6.4%` vs 2b) | `49.60%` | **`55.60%`** (`+6.0%` vs 3a) |
| **`InDom-500` xLAM (Clean Types)** | `250` | `69.60%` | `82.80%` | **`83.60%`** | `80.80%` | **`82.40%`** | `79.20%` | **`82.80%`** |
| **`InDom-500` ToolACE (Quoted Ints)** | `250` | `34.80%` | `46.40%` | **`47.60%`** | `39.20%` | `42.80%` | `38.80%` | **`43.60%`** |

1. **Label Artifacts in `RLLA-80` (`16 / 80` Noisy GTs)**: Exactly 20% (`16/80`) of the original `RLLA-4k` test set contains corrupted Ground Truth annotations—either truncated unclosed JSON strings (e.g., `"[{"subject": "Math"`) or numbers wrapped in string quotes contradicting `type: integer` in the schema. `Pure SFT` memorizes these dataset-specific bugs from `train.parquet` (`50.0%` match on broken GTs), whereas `Pure RL` and `Cold-Start SFT + GRPO` always output syntactically valid, schema-typed JSON (`0.0%` on broken GTs).
2. **Near-Parity on Clean In-Domain Schemas (`xLAM-250`)**: On the `250` unseen in-domain queries from `xLAM` (where Ground Truth strictly follows JSON Schema types), **`Pure RL (v2)` (`82.80%`) and `Cold-Start SFT + GRPO` (`82.40%`) virtually tie `Pure SFT` (`83.60%`)** without memorizing training strings!
3. **Where Pure RL Beats Pure SFT Even on In-Domain Queries**: On tasks requiring **implicit arithmetic or interval reasoning** prior to parameter filling—such as computing zero-indexed pagination `skip = 20 - 1 = 19` (`rlla_test_20`) or inclusive count `80 - 70 + 1 = 11` (`rlla_test_55`)—`Pure RL` and `Cold-Start SFT + GRPO` achieve `1.0` Exact Match via `<think>` Chain-of-Thought, while `Pure SFT` blindly copies literal numbers (`20`, `10`) from the prompt (`0.0`).

---

## 4. Quantitative Error Taxonomy (`E1`–`E7`) & Failure Case Analysis

To understand *how* each model fails rather than just reporting aggregate accuracy, `eval/analyze_error_taxonomy.py` classifies every non-exact prediction across all benchmarks into **7 mutually exclusive error categories**:
- **`E1` Format / JSON Syntax Error**: Missing `<tool_call>` XML wrapper or unparseable JSON payload.
- **`E2` Mode / Abstention Error**: Falsely refusing to call a tool (`<response>`) when a valid tool exists, or invoking a tool when abstention is required.
- **`E3a` Wrong Tool Selection**: Selecting an incorrect tool name when call count matches.
- **`E3b` Tool Call Count Error**: Emitting too few or too many parallel tool calls (e.g., calling 1 tool instead of 2).
- **`E4` Missing Required Parameter Key**: Omitting a mandatory parameter specified in the tool schema.
- **`E5` Hallucinated Extra Parameter Key**: Inventing or redundantly filling optional parameters not requested by the user.
- **`E6` Parameter Value / Arithmetic Error**: Correct tool and keys, but wrong argument value (e.g., off-by-one arithmetic or string mismatch).
- **`E7` Type / Quote Mismatch Only**: Semantic value is correct (`Schema-Normalized EM = 1.0`), but JSON data type differs (e.g., `"10"` vs `10`).

<div align="center">
  <img src="assets/fig4_reward_v2_staircase.png" width="98%" alt="Reward v2 Staircase & Quantitative Error Taxonomy"/>
</div>

### Error Taxonomy Breakdown Across Benchmarks (`eval/error_taxonomy_report.json`)

#### A. Out-of-Domain `BFCL-v3 Standard` (`N = 1,000` Tool-Calling Queries)

| Model | `CORRECT` (Strict EM) | `E1` Format/JSON | `E2` False Abstain | `E3a` Wrong Tool | `E3b` Call Count | `E4` Missing Key | `E5` Extra Key | `E6` Value Error | `E7` Type/Quote Only |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`0. Zero-Shot Base`** | `695` (`69.5%`) | `8` | `2` | `8` | `150` | `4` | `0` | `102` | `31` |
| **`1a. Pure SFT (1.0 Ep)`** | `602` (`60.2%`) | `6` | `0` | `2` | `32` | `69` | `2` | `141` | **`146`** |
| **`1b. Pure SFT (2.0 Ep)`** | `604` (`60.4%`) | `14` | `2` | `3` | `28` | `52` | `2` | `158` | **`137`** |
| **`2a. Full-SFT + GRPO`** | `595` (`59.5%`) | `14` | `2` | `3` | `25` | `55` | `2` | `164` | **`140`** |
| **`2b. Cold-Start SFT (320)`** | `793` (`79.3%`) | `7` | `1` | `7` | `45` | `6` | `0` | `118` | `23` |
| **`2c. Cold-Start SFT + GRPO`** | **`812` (`81.2%`)** | **`4`** | **`0`** | **`6`** | `27` (`-40%` vs 2b) | **`6`** | `1` | `126` | **`18`** |
| **`3a. Pure RL (4.0 Ep, v1)`** | `797` (`79.7%`) | `5` | `2` | `8` | `28` | `6` | `0` | `130` | `24` |
| **`3b. Pure RL (8.0 Ep, v2)`** | **`815` (`81.5%`)** | `5` | `2` | `7` | **`18`** (`-36%` vs 3a) | `8` | `1` | **`125`** | `19` |

#### B. Out-of-Domain `BFCL-v3 Live` (`N = 1,351` Real-World Multi-Candidate Queries)

| Model | `CORRECT` (Strict EM) | `E1` Format/JSON | `E2` False Abstain | `E3a` Wrong Tool | `E3b` Call Count | `E4` Missing Key | `E5` Extra Key | `E6` Value Error | `E7` Type/Quote Only |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`0. Zero-Shot Base`** | `781` (`57.8%`) | `3` | `3` | `57` | `21` | `32` | `17` | `357` | `80` |
| **`1a. Pure SFT (1.0 Ep)`** | `712` (`52.7%`) | `14` | `5` | `27` | `22` | `86` | `10` | `293` | **`182`** |
| **`1b. Pure SFT (2.0 Ep)`** | `666` (`49.3%`) | `11` | **`49` (`9.8×` surge)** | `25` | `21` | `66` | `11` | `318` | **`184`** |
| **`2c. Cold-Start SFT + GRPO`** | **`797` (`59.0%`)** | **`1`** | **`0`** | **`47`** | `27` | `48` | **`13`** | `334` | `84` |
| **`3b. Pure RL (8.0 Ep, v2)`** | `794` (`58.8%`) | **`1`** | `3` | `50` | `29` | `49` | `15` | **`318`** | `92` |

#### C. Strictly In-Domain `Expanded In-Domain 500` (`N = 500` Unseen Queries)

| Model | `CORRECT` (Strict EM) | `E1` Format/JSON | `E2` Mode Error | `E3a` Wrong Tool | `E3b` Call Count | `E4` Missing Key | `E5` Extra Key | `E6` Value Error | `E7` Type/Quote Only |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`0. Zero-Shot Base`** | `261` (`52.2%`) | **`235` (No XML)** | `0` | `0` | `0` | `0` | `0` | `0` | `4` |
| **`1b. Pure SFT (2.0 Ep)`** | **`328` (`65.6%`)** | `3` | `5` | `1` | `15` | `35` | `14` | **`78`** | `21` |
| **`2b. Cold-Start SFT (320)`** | `300` (`60.0%`) | `15` | `1` | `3` | `29` | `7` | `49` | `93` | `2` |
| **`2c. Cold-Start SFT + GRPO`** | `313` (`62.6%`) | `15` | `1` | `3` | `21` (`-28%`) | `9` | `39` (`-20%`) | `97` | `2` |
| **`3a. Pure RL (4.0 Ep, v1)`** | `295` (`59.0%`) | `16` | `1` | `6` | `21` | `8` | `50` | `100` | `3` |
| **`3b. Pure RL (8.0 Ep, v2)`** | `316` (`63.2%`) | `16` | `0` | `4` | `19` | `11` | **`36` (`-28%`)** | `98` | **`0`** |

### Three Key Insights from the Error Taxonomy
1. **Why SFT Collapses on OOD (`E7` Type Contamination + `E4` Missing Keys + `E2` False Abstention)**:
   - On `BFCL-v3 (1k)`, `Pure SFT` commits **`137–146` `E7` Type/Quote errors** (`7.6×` more than `Pure RL`'s `18–19` cases) and **`52–69` `E4` Missing Required Key errors** (`8×` more than `Pure RL`'s `6–8` cases). Without `<think>` verification, SFT skips mandatory parameters and quotes integers/booleans as strings.
   - On `BFCL-Live (1,351)`, over-training SFT from `1.0 Ep` to `2.0 Ep` causes **`E2` False Abstention errors to jump from `5` to `49` (`9.8×`)**, explaining the `-3.40%` OOD drop!
2. **How `v2` Anti-Hacking Reward Improves Pure RL (`E5` & `E3b` Reduction)**:
   - Upgrading from `rlla_reward.py (v1)` to `rlla_reward_v2.py (v2)` reduces **`E5` Hallucinated Extra Parameter Keys** on `In-Domain 500` from `50` $\rightarrow$ `36` (`-28%`) and reduces **`E3b` Tool Call Count Errors** on `BFCL-v3` from `28` $\rightarrow$ `18` (`-36%`).
3. **How GRPO Improves Cold-Start SFT (`2b` $\rightarrow$ `2c`)**:
   - Adding GRPO (`v2`) on top of `Cold-Start SFT (320)` cuts `E3b` Multi-Tool Count Errors by **`-40%` (`45` $\rightarrow$ `27`)** on `BFCL-v3` and cuts `E5` Extra-Key Errors by **`-20%` (`49` $\rightarrow$ `39`)** on `In-Domain 500`, proving that GRPO actively refines multi-tool composition and parameter precision when policy entropy is preserved.

---

## 5. Conclusion: Comprehensive Comparison of SFT vs. RL — Advantages, Disadvantages, and Trade-Offs

Synthesizing all 8 controlled models across `3,171` test instances and the `E1`–`E7` error taxonomy provides a balanced, evidence-backed answer to **how SFT and RL differ and what each paradigm gains and sacrifices**:

### 5.1 Mechanistic Comparison: Surface Behavior Cloning (`SFT`) vs. Sequence Outcome Exploration (`RL`)

| Dimension | Supervised Fine-Tuning (`Pure SFT`) | Reinforcement Learning (`Zero-RL GRPO` / `Cold-Start SFT + GRPO`) |
| :--- | :--- | :--- |
| **Optimization Signal** | **Token-Level Cross-Entropy (Teacher Forcing)**: Imitates literal reference tokens; a dropped key at the end carries only $1/T$ weight. | **Sequence-Level Outcome Verification**: Scores the final `<tool_call>` AST holistically (`+4.0` strictly requires 100% exact match). |
| **Core Acquired Skill** | **Memorizes Seen API Conventions & Refusal Boundaries**: Rapidly fits seen tool names, default habits, refusal patterns, and even label bugs. | **Learns Dynamic Schema Reasoning**: Inspects unseen JSON Schemas in the prompt, checks required types, and computes parameters. |
| **Effect on `<think>` CoT** | Overwrites pretrained reasoning (`0.0%` genuine CoT) when training data contains placeholder `<think>` strings. | Applies zero text loss to `<think>`, naturally preserving **`100.0%` genuine multi-step CoT** because reasoning improves call accuracy. |
| **Compute & Latency** | **8 GPUs, 5–10 mins** (`0.7–1.3 GPU-hrs`); concise outputs (`~328` chars), low online latency. | **32 GPUs, 32–75 mins** (`17–40 GPU-hrs`); longer CoT outputs (`~510` chars), higher inference latency. |

### 5.2 Four Core Advantages of RL Over SFT (Where RL Excels)

1. **Superior Out-of-Domain Generalization to Unseen Tools (`+21.10%` on `BFCL-v3`)**:
   On `1,000` unseen OOD tool-calling tasks (`BFCL-v3 Standard`), Full-SFT (`60.20%–60.40%`) fails to improve over the base model, whereas **`Pure RL` reaches `81.50%` Strict EM (`+21.10%` lead)** and leads by **`+6.3%–9.7%` on `BFCL-Live` (`1,351` queries)**—cutting `E7` type/quote errors by **`87%`** and `E4` missing-required-key errors by **`85%`**.
2. **Compositional Multi-Tool Planning & Implicit Arithmetic Reasoning**:
   Even within seen In-Domain tools, whenever arguments require arithmetic derivation (0-indexed pagination `skip = 20 - 1 = 19`, inclusive intervals `80 - 70 + 1 = 11`) or complex parallel composition (`BFCL Parallel-Multiple`: **`76.50%` RL vs. `41.50%` SFT, `+35.0%` lead**), RL succeeds via genuine `<think>` planning and sequence-level credit assignment.
3. **Immunity to Imperfect / Boilerplate Supervision**:
   Despite `99.36%` of `RLLA-4k` containing a static boilerplate `<think>` and `20%` containing quoting/truncation noise, outcome-based RL ignores flawed `<think>` text, retains **`100%` genuine reasoning**, and achieves **`82.80%` on clean In-Domain `xLAM-250`** (virtually tying Full-SFT's `83.60%`).
4. **Label-Free Self-Evolution on Unannotated Traffic (No-GT Scalability)**:
   Unlike SFT, RL can continuously improve on raw production queries using **JSON Schema validation + sandbox dry-run execution outcomes** without human Ground Truth labels.

### 5.3 Four Objective Disadvantages & Limitations of RL Compared to SFT (Where RL Falls Short)

Our experiments equally highlight four concrete limitations where **RL incurs substantial costs or underperforms SFT**:

1. **High Training Compute & Wall-Clock Cost (`15×–60×` More GPU-Hours)**:
   `Pure SFT (1.0 Ep)` converges in **5 minutes** on 8 GPUs (`0.67 GPU-hrs`), whereas `Pure RL (8.0 Ep)` requires **75 minutes** on 32 GPUs (`40 GPU-hrs`; `~17 GPU-hrs` for `Cold-Start SFT + GRPO`) due to online $G=8$ rollout generation at every step.
2. **Higher Online Inference Latency & Token Cost (`~1.6×` Output Length)**:
   RL's accuracy gains rely on `~270–280` characters of explicit `<think>` reasoning (`~510` total chars vs. `~328` chars for SFT), which poses a latency bottleneck under strict real-time budgets (e.g., `<300ms` voice or vehicle-control gateways).
3. **Weaker Irrelevance Abstention & Over-Triggering Bias (`BFCL Irrelevance`: `15.8%–22.5%` RL vs. `76.3%–89.2%` SFT)**:
   On `BFCL Irrelevance (240 queries where no candidate tool applies)`, **`Pure SFT` reaches `76.25%–89.17%`, whereas `Pure RL / Cold-Start SFT+RL` scores only `15.83%–22.50%`**. Because abstention examples comprise only ~11% of `RLLA-4k` and dense partial rewards incentivize attempting tool calls, RL exhibits a strong **action bias (over-triggering)** unless negative no-tool samples and refusal penalties are explicitly rebalanced.
4. **High Sensitivity to Reward Hacking & Initial Policy Entropy**:
   SFT optimizes a stable cross-entropy objective out of the box; RL immediately exploits minor reward loopholes (e.g., `50` `E5` optional-key stuffing errors in `v1`) and stalls completely if preceded by heavy SFT that collapses policy entropy (`0.085` in `Full-SFT + GRPO`).

### 5.4 Practical Engineering Selection Matrix

| Scenario & Constraints | Recommended Recipe | Trade-Off Rationale |
| :--- | :--- | :--- |
| **Closed, Static Internal APIs + Strict Latency + High Refusal Precision** | **`Early-Stopped Pure SFT (1.0 Ep, Min Eval-Loss)`** | Leverages SFT's **5-min training cost, low latency (`~328` chars), and strong refusal boundary (`76.3%`)** while avoiding Epoch 2 OOD degradation. |
| **Open-World / MCP Tool Ecosystems + Compositional / Arithmetic Reasoning** | **`Pure RL (Zero-RL GRPO, v2 Anti-Hacking Reward)`** | Maximizes **unseen-schema generalization (`BFCL-v3 81.5%`) and `100%` genuine CoT** (augment no-tool negative samples if high abstention is needed). |
| **Enterprise Multi-Candidate Assistants (Fast RL Convergence + Live Robustness)** | **`Cold-Start SFT (320 Genuine-CoT, 1 Ep) + GRPO (v2)`** | Uses a 1-min SFT warm-up to establish format compliance (`93.8%`) while preserving entropy (`0.492`), reaching peak validation reward (`+2.7486`) and best **`BFCL-Live` accuracy (`58.99%`)**. |

---

## 6. Repository Structure & Quick Reproduction

```text
agent-rl-lab/
├── README.md                                  # English primary documentation
├── README_zh.md                               # Chinese documentation (简体中文)
├── assets/                                    # 300-DPI bilingual figures (fig1..4.png & fig1..4_zh.png)
├── configs/
│   ├── sft_rlla_qwen3_4b_ep2.yaml             # Stage 1: Full-SFT config (3,920 samples, eval every epoch)
│   └── sft_coldstart_320_qwen3_4b_ep1.yaml    # Stage 3: Cold-Start SFT config (320 genuine-CoT samples, 1 Ep)
├── data/
│   ├── rlla_4k/                               # RLLA-4k train (3,920), test (80), & test_indomain_500.parquet (500)
│   └── sft_coldstart_320/                     # 320 rejection-sampled genuine-CoT trajectories for Cold-Start SFT
├── docs/
│   ├── SFT_vs_RL_Tool_Use_Deep_Dive.md        # Comprehensive technical report (English)
│   └── SFT_vs_RL_Tool_Use_Deep_Dive_zh.md     # Comprehensive technical report (Chinese)
├── eval/
│   ├── run_multi_benchmark_eval.py            # Evaluator for RLLA-4k (80) + BFCL-v3 Standard (1,240)
│   ├── run_expanded_indomain_eval.py          # Evaluator for Expanded In-Domain Test Set (500)
│   ├── run_bfcl_live_eval.py                  # Evaluator for BFCL-v3 Live (1,351)
│   ├── summarize_multi_benchmark.py           # Unified 8-model x 4-suite report generator
│   ├── analyze_error_taxonomy.py              # 7-category Error Taxonomy (E1-E7) analyzer
│   ├── multi_benchmark_report.json            # Full machine-readable metrics for all 8 models
│   └── error_taxonomy_report.json             # Full machine-readable error taxonomy counts
├── rewards/
│   ├── rlla_reward.py                         # Stage 2a: Initial dense tool-calling reward function (v1)
│   └── rlla_reward_v2.py                      # Stage 2b & 3: Anti-reward-hacking reward function (v2)
└── scripts/
    ├── prepare_model_and_data.py              # Vocabulary strip & RLLA-4k Parquet/ShareGPT preprocessing
    ├── build_expanded_indomain_test.py        # Builds 500-sample AST-verified strictly In-Domain test set
    ├── run_sft_rlla.sh                        # Launch LLaMA-Factory SFT
    ├── run_grpo_rlla.sh                       # Launch multi-node VeRL GRPO (v1 reward)
    ├── run_grpo_rlla_v2_8ep.sh                # Launch multi-node VeRL GRPO (v2 reward, Pure RL & Cold-Start SFT+RL)
    └── generate_github_figures.py             # Re-generate all English & Chinese figures from logs/metrics
```

### Quick Start Commands

```bash
# 1. Prepare Clean Base Model & Datasets (including 500-sample Expanded In-Domain Test Set)
python scripts/prepare_model_and_data.py
python scripts/build_expanded_indomain_test.py

# 2. Stage 1: Run Full SFT (select checkpoint-62 @ 1.0 Ep for min eval_loss, or checkpoint-124 @ 2.0 Ep)
bash scripts/run_sft_rlla.sh

# 3. Stage 2: Run Pure RL (Zero-RL GRPO with v1 -> v2 Anti-Reward-Hacking)
bash scripts/run_grpo_rlla.sh
bash scripts/run_grpo_rlla_v2_8ep.sh

# 4. Stage 3: Run Cold-Start SFT (320 Genuine-CoT, 1 Ep) + GRPO (v2 Reward)
llamafactory-cli train configs/sft_coldstart_320_qwen3_4b_ep1.yaml
INIT_MODEL=outputs/sft/sft-coldstart320-Qwen3-4B-Base-rlla-ep1 bash scripts/run_grpo_rlla_v2_8ep.sh

# 5. Evaluate All 8 Models Across All 4 Benchmark Suites (N = 3,171) & Generate Error Taxonomy
python eval/summarize_multi_benchmark.py
python eval/analyze_error_taxonomy.py
python scripts/generate_github_figures.py
```

---

## License

This project is released under the [Apache-2.0 License](LICENSE).
