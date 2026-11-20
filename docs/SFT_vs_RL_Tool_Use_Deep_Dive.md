# SFT Memorization vs. RL Generalization in Tool Use: An Empirical Study on RLLA-4k and BFCL-v3

**English** | [简体中文](SFT_vs_RL_Tool_Use_Deep_Dive_zh.md)

## Abstract

When building tool-calling (Function Calling) models, engineering teams routinely face three practical questions:
1. If the objective is to maximize accuracy on a fixed benchmark or a fixed set of vertical-domain APIs, is Supervised Fine-Tuning (SFT) more efficient than Reinforcement Learning (RL)?
2. Why does applying GRPO on top of a full-dataset SFT checkpoint frequently result in flat reward curves?
3. What is the concrete value of RL within a fixed vertical domain—both with `<think>` Chain-of-Thought (CoT) and without `<think>`—and how can rewards be computed in production environments without Ground Truth (GT) labels?

Using **Qwen3-4B-Base**, we align **LLaMA-Factory (SFT)** and **VeRL (GRPO)** at the token level, train five controlled model variants on the same 3,920-sample `ToolRL / RLLA-4k` dataset, and evaluate them across **2,431 test problems** spanning **RLLA-4k (80 in-domain samples)**, **BFCL-v3 Standard (1,000 out-of-domain samples)**, and **BFCL-v3 Live (1,351 real-world user queries)**.

---

## 1. Experimental Setup & Headline Results

All trained models see only the 3,920 training prompts in `RLLA-4k` and are evaluated zero-shot across three distinct test distributions:

- **RLLA-4k Test (`N=80`, In-Domain)**: Drawn from the exact same source pool as the training set, sharing identical tool schemas, default parameter conventions, and upstream preprocessing artifacts.
- **BFCL-v3 Standard (`N=1,000`, Out-of-Domain)**: The UC Berkeley Function Calling Leaderboard benchmark (`Simple 400`, `Multiple 200`, `Parallel 200`, `Parallel-Multiple 200`). None of the test APIs appear in the training set.
- **BFCL-v3 Live (`N=1,351`, Out-of-Domain Real-World)**: Real user queries contributed by developers, featuring long prompts and 10–20 closely related candidate tools per query.

![In-Domain Memorization (SFT) vs. Out-of-Domain Generalization (RL)](../assets/fig1_sft_vs_rl_overview.png)

| Model Configuration | Compute & Time | `RLLA-4k` In-Domain<br>(All 80 / Clean 64) | `BFCL-v3` Standard OOD<br>(N=1,000 · Strict / Norm EM) | `BFCL-Live` Real-World OOD<br>(N=1,351 · Strict / Norm EM) | Genuine CoT Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **0. Zero-Shot Base** (`Qwen3-4B-Base-rlla`) | — | `43.75%` / `54.69%` | `60.20%` / `63.60%` | `45.30%` / `51.89%` | `93.8%` |
| **1. Pure SFT** (`3,920 samples × 2 Epochs`) | 8× A800 · 10 mins | **`73.75%`** / **`79.69%`** | `60.40%` / `64.60%` | `49.30%` / `58.92%` | `0.0%` |
| **2. Full-SFT + RL** (`2Ep SFT + 120-step GRPO`) | 10m + 1.5 hrs | `72.50%` / `78.12%` | `59.70%` / `63.90%` | `48.93%` / `58.77%` | `0.0%` |
| **3. Cold-Start SFT** (`320 samples × 1 Epoch`) | 8× A800 · 1 min | `50.00%` / `60.94%` | `67.10%` / `70.50%` | `51.81%` / `59.44%` | `95.0%` |
| **4. Pure RL** (`Zero-RL GRPO 4 Epochs, v1`) | 32× A800 · 1.5 hrs | `56.25%` / `70.31%` | `79.70%` / `81.90%` | `58.11%` / `64.99%` | `97.5%` |
| **5. Pure RL** (`GRPO 8 Epochs, v2 Reward`) | 32× A800 · 3.0 hrs | `57.50%` / **`71.88%`** | **`81.50%`** / **`83.40%`** | **`58.77%`** / **`65.58%`** | **`98.8%`** |

### Key Takeaways from the Benchmark Table
1. **For a fixed in-domain test set (`RLLA-4k`), SFT is fastest and achieves the highest raw score**:
   - Pure SFT reaches `73.75%` (`79.69%` on clean samples) in **10 minutes** on 8 GPUs (18× faster than RL). Token-level cross-entropy directly memorizes domain-specific parameter conventions and refusal templates.
2. **For out-of-domain generalization (`BFCL-v3` & `BFCL-Live`), RL substantially outperforms SFT**:
   - On the 1,000 unseen tool-calling problems in `BFCL-v3`, Pure SFT (`60.40%`) barely improves over the untrained Base model (`60.20%`).
   - Pure RL (8 Epochs) reaches **`81.50%`** (`83.40%` schema-normalized EM), outperforming Pure SFT by **+21.1 percentage points**, and by **+9.5 percentage points** on `BFCL-Live` (`58.77%` vs. `49.30%`).

---

## 2. In-Domain vs. Out-of-Domain Discrepancy: Auditing `RLLA-4k` Label Quality

To understand why Pure RL achieves `81.50%` on `BFCL-v3` but `57.50%` on the raw 80-sample `RLLA-4k` test set, we audited all 3,920 training rows and 80 test rows in `RLLA-4k`.

![In-Domain Label Quality Audit & Out-of-Domain Complexity Scaling](../assets/fig3_deep_dive_clean_dirty_and_bfcl.png)

### 2.1 Upstream Conversion Artifacts in `RLLA-4k`
`RLLA-4k` was constructed by converting ToolACE, xLAM, and Hammer. During conversion, nested parameter strings were split using a naive `.split(",")` heuristic, corrupting **16 out of 80 test samples (20%)**:
1. **Truncated Unclosed JSON Strings (6 test samples, 536 training samples)**:
   For example, in Test Sample 18, the ground-truth parameter value is truncated mid-JSON:
   ```json
   {"name": "create_student_profile", "parameters": {"courses": "[{\"subject\": \"Mathematics\""}}
   ```
2. **Schema-Contradicting String Quotes (10 test samples, 945 training samples)**:
   Parameters explicitly typed as `integer` or `boolean` in the tool's JSON Schema are labeled as strings (`"1"`, `"True"`) in the Ground Truth.

### 2.2 Performance on Clean vs. Corrupted Subsets
- **On the 16 corrupted test samples**: Pure RL scores **`0.00%` (0/16)** because it emits syntactically valid, closed JSON that fails string-exact comparison against truncated Ground Truths. Pure SFT scores **`50.00%` (8/16)** because token-level teacher forcing memorized the 536 analogous truncation bugs in the training split.
- **On the 64 clean test samples**: Pure RL accuracy rises to **`71.88%` (46/64)** vs. Pure SFT's `79.69%` (51/64).

---

## 3. Role of RL Within a Fixed Vertical Domain

If a model is deployed strictly within a closed vertical domain (where out-of-domain benchmarks like `BFCL` are irrelevant), what does RL contribute? Inspecting the in-domain `RLLA-4k` samples where **Pure RL succeeds (`1.0`) while Pure SFT fails (`0.0`)** reveals two distinct regimes:

### 3.1 With `<think>` CoT: Solving Implicit Arithmetic & Multi-Step Parameter Derivation
While ~80% of vertical-domain queries require direct entity copying (where SFT excels), 15%–20% require arithmetic or relational derivation before filling parameter slots:

- **Case 1: Pagination Offset Derivation (`rlla_test_20`)**
  - **User Query**: *"Fetch the 20th to 30th product from the catalog"* using `get_products(skip, limit)`.
  - **Pure SFT (Incorrect)**: Copies the surface number `20` directly into `{"limit": 10, "skip": 20}` (which skips 20 items and starts at the 21st product).
  - **Pure RL (Correct)**: Computes `20 - 1 = 19` inside `<think>` (*"skip the first 19 products and limit the response to 10 products"*) and emits `{"skip": 19, "limit": 10}`.
- **Case 2: Inclusive Range Counting (`rlla_test_55`)**
  - **User Query**: *"Fetch ticker data from index 70 up to 80 inclusive."*
  - **Pure SFT (Incorrect)**: Emits count `10`.
  - **Pure RL (Correct)**: Derives `80 - 70 + 1 = 11` inside `<think>` (*"fetch 11 tickers"*) and emits `11`.

### 3.2 Without `<think>` (Direct `<tool_call>` Generation)
When `<think>` is removed to minimize online latency, RL loses its scratchpad for arithmetic derivation, and data-centric SFT should be the primary focus. Without `<think>`, RL provides targeted value in three areas:
1. **Sequence-Level Penalty for Partial Omissions / Hallucinated Keys**: Token-level SFT loss treats omitting the parameters of the 3rd tool in a 3-tool call (e.g., `rlla_test_34`) as a minor 5% token error, whereas RL penalizes the entire trajectory unless all tool calls are complete.
2. **Calibrating Tool-Call vs. Clarification Thresholds**: Adjusting abstention vs. invocation rewards directly shifts the model's conservatism when required parameters are ambiguous.
3. **Training on Unlabeled Production Queries**: Leveraging rule-based schema verifiers on production traffic where Ground Truth annotations do not exist.

---

## 4. Why Full-SFT + GRPO Stalls: Boilerplate CoT & Entropy Collapse

In `Model 2 (Full-SFT + RL)`, running 150 steps of GRPO after 2 epochs of full-dataset SFT produced a flat validation reward curve (`+2.9558 -> +2.9370`).

![Training Dynamics: Pure RL vs. Full-SFT + GRPO](../assets/fig2_training_dynamics_entropy_collapse.png)

### 4.1 Divergent Supervision Mechanics on `<think>`
Every row in `RLLA-4k` contains the same boilerplate sentence inside `<think>`:
```xml
<think> I should use the appropriate tool with proper parameters to respond to the user's need. </think>
```
- **SFT (Token-Level Supervision)** forces the model to predict every token of this boilerplate `<think>` across 3,920 samples, driving genuine CoT to `0.0%` and collapsing policy entropy (`actor/entropy`) from `0.72` to `0.08`.
- **Pure RL (Outcome Supervision)** never exposes the ground-truth `<think>` to the policy; `compute_score` discards `<think>` and evaluates only the `<tool_call>` JSON block. Consequently, genuine reasoning trajectories generated by the Base model receive positive reinforcement (`98.8%` genuine CoT).

### 4.2 Zero Group Variance in GRPO
At `actor/entropy = 0.08`, all $G=8$ rollouts sampled for a prompt are identical, yielding $\text{std}(r_1, \dots, r_G) = 0$ and zero advantage gradients ($A_i = 0$). Across 120 GRPO steps, 75 of 80 validation responses remained character-for-character identical to Step 0.

---

## 5. Reward Design (`v2`) & Training Without Ground Truth

### 5.1 Dense Shaping & Anti-Reward-Hacking (`rewards/rlla_reward_v2.py`)

![Dense Reward Shaping & Anti-Reward-Hacking Design](../assets/fig4_reward_v2_staircase.png)

To prevent zero-gradient plateaus on multi-tool problems while mitigating reward hacking, `rlla_reward_v2.py` introduces:
1. **Extra-Key Penalty (`-0.75 / key`)**: Deducts `0.75` for each predicted parameter key absent from the Ground Truth, preventing the policy from guessing optional schema fields.
2. **Non-Exact Match Cap (`+3.4` vs. `+4.0`)**: Caps partial matches at `+3.4` total score so that 100% strict exact match (`+4.0`) always maintains a `+0.60` advantage jump.

### 5.2 Reward Construction Without Ground Truth in Production
When scaling RL to unlabeled production queries where no human Ground Truth exists, the verifier can be constructed as a 3-tier pipeline:
1. **Tier 1 — JSON Schema & Dry-Run Sandbox Verifier**: Validates syntax, tool existence, `required` keys, `enum` constraints, and parameter types directly against the tool's JSON Schema (zero labeling cost).
2. **Tier 2 — Execution Outcome Assertions**: Rewards non-empty returns or passing post-conditions in read-only/sandbox environments.
3. **Tier 3 — LLM-as-a-Judge Semantic Verification**: Uses a larger judge model offline to verify binary parameter semantics (e.g., verifying whether `skip: 19` matches *"20th to 30th item"*), which is substantially more reliable than open-ended synthetic data generation.
