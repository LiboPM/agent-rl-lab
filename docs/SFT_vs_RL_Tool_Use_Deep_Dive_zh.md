# 大模型工具调用（Tool Use）中 SFT 与 RL (GRPO) 的三阶段系统性实证报告

<div align="center">

[![返回中文主页](https://img.shields.io/badge/主页-README_中文版-059669?style=for-the-badge)](../README_zh.md)
[![English Report](https://img.shields.io/badge/Report-English_Version-2563eb?style=for-the-badge)](SFT_vs_RL_Tool_Use_Deep_Dive.md)

</div>

---

## 1. 核心摘要与三阶段递进式实验设计

本技术报告基于 `Qwen3-4B-Base`，在 4 节点 × 8 卡 NVIDIA A800-80GB（共 32 卡）集群上，对工具调用（Function Calling / Tool Use）后训练中的三大核心范式 —— **监督微调（`Pure SFT`）**、**直接强化学习（`Zero-RL GRPO`）** 与 **先监督微调再强化学习（`SFT + GRPO`）** —— 开展了严格受控的三阶段递进式实证研究：

1. **第一阶段 —— 监督微调（`Pure SFT`）探索与过拟合拐点定位**：
   - 在 `RLLA-4k` 全量训练集（`3,920` 条）上训练 2 个 Epoch（`124` 步），完整追踪 `train_loss` 与验证集 `eval_loss`。
   - 实验发现：验证集 `eval_loss` 在 **`1.0 Epoch`（`Step 62`，`eval_loss = 0.1582`）达到全局最低点**；继续训练至 **`2.0 Epochs`（`Step 124`）** 时，虽然 `train_loss` 从 `0.1510` 进一步降至 `0.0196`，但验证集 `eval_loss` 反弹至 `0.1700`（恶化 `+7.5%`）。
   - 跨基准评测证实：从 `1.0 Ep` 到 `2.0 Ep` 仅在完全共享训练集 API 的 `扩充同分布 500 题` 上微增 `+1.0%`（`64.60%` $\rightarrow$ `65.60%`），却在分布外真实企业级基准 `BFCL-Live (1,351题)` 上导致严格精确匹配率 **下跌 `-3.40%`（`52.70%` $\rightarrow$ `49.30%`）**，原因是过度拒答错误（`E2`）暴增了 **`9.8 倍`（`5` $\rightarrow$ `49` 例）**。同时，`RLLA-4k` 中占比 `99.4%` 的 `<think>` 伪模板句使 SFT 模型的真实推理链保留率归零（`0.0%`）。
2. **第二阶段 —— 纯强化学习（`Zero-RL GRPO`）探索、反作弊奖励设计（`v1` $\rightarrow$ `v2`）与收敛峰值选择**：
   - 直接从基座模型启动 **VeRL GRPO**（`v1` 奖励），在 `Step 120`（`4.0 Epochs`）彻底消除了基座的格式坍缩（从 `97.5%` 降至 `2.5%`），`100.0%` 保留了真实 `<think>` 推理链，并在分布外 `BFCL-v3 (1,000题)` 上达到 **`79.70%` 严格 EM**。
   - 通过错误分类学（Error Taxonomy）诊断 `Model 3a (v1)` 发现：由于 `v1` 对多填冗余可选参数仅扣 `-0.25/key` 且未对非全对得分封顶，模型在 `扩充同分布 500 题` 上出现了 `50` 例 **`E5_Hallucinated_Extra_Param_Key`（投机性多填可选参数）**。
   - 升级至反作弊奖励函数 `rewards/rlla_reward_v2.py`（冗余参数惩罚加重至 `-0.75/key`，非全对总分封顶 `+3.40` vs 全对 `+4.00`）并继续训练，验证集奖励在 **`Step 120`（累计 `8.0 Epochs`）达到全局峰值 `+2.7421`**（`Step 150` 小幅回落至 `+2.7077`）。选取 `Step 120` 峰值 Checkpoint（`Model 3b`），`E5` 多填参数错误下降 **`28%`（`50` $\rightarrow$ `36` 例）**，扩充同分布 500 题严格 EM 提升 **`+4.20%`（`59.00%` $\rightarrow$ `63.20%`）**，`BFCL-v3 标准集` 达到 **`81.50%` 严格 EM（`83.40%` 归一化 EM，领先 SFT `+21.10%`）**！
3. **第三阶段 —— `SFT + GRPO` 组合探索：破解策略熵坍缩悖论（负向对照 vs. 正向对照）**：
   - **负向对照（`Model 2a: Full-SFT 2Ep + GRPO`）**：以策略熵已坍缩至 `0.085` 的全量 SFT 模型作为起点跑 120 步 GRPO，组内 $G=8$ 次采样输出高度同质化（$\sigma_r = 0 \implies A_i = 0$），验证集奖励完全停滞（`+2.9558` $\rightarrow$ `+2.9370`，`75/80` 题输出与 Step 0 逐字相同）。
   - **正向对照（`Model 2c: Cold-Start SFT 320 + GRPO`）**：使用拒绝采样筛选的 **`320` 条含真实 `<think>` 推理链的高质量样本** 仅做 `1.0 Epoch`（`5` 步更新）微缩冷启动（`Model 2b`），在将初始格式合规率拉升至 `93.75%`（`Step 0` 验证集奖励 `+2.5112`）的同时，将初始策略熵完好保留在 **`0.492`**。随后接入 `GRPO (v2 奖励)`，验证集奖励稳步攀升并在 **`Step 100`（`3.33 Epochs`）达到全局峰值 `+2.7486`**，使 `扩充同分布 500 题` 提升 **`+2.60%`**（多工具并发提升 **`+6.40%`**）、`BFCL-v3 标准集` 提升 **`+1.90%`**（`Parallel-Multiple` 提升 **`+5.50%`**），并在真实企业级 `BFCL-Live (1,351题)` 上取得全场最高的 **`58.99%` 严格 EM**！

<div align="center">
  <img src="../assets/fig1_sft_vs_rl_overview_zh.png" width="96%" alt="四大域内与域外基准五大代表模型严格精确匹配率对比"/>
</div>

---

## 2. 四大评测基准构建与 500 题严格同分布扩充集说明 (`N = 3,171`)

原 `RLLA-4k` 测试集（`data/rlla_4k/test.parquet`）仅有 `80` 道样本，且其中 `16` 道题（占比 `20%`）存在 Ground Truth JSON 字符串截断或整型加引号的标注缺陷。为了彻底消除小样本统计波动并客观评估模型在 **同分布（In-Domain）** 与 **分布外（Out-of-Domain, OOD）** 上的真实水平，我们编写了 `scripts/build_expanded_indomain_test.py`，构建了 `data/rlla_4k/test_indomain_500.parquet`（`N = 500`）：

1. **100% 严格同分布工具覆盖**：从 `ToolACE` 与 `xLAM-60k` 上游语料中筛选出的这 `500` 道测试题，其 System Prompt 中的 **每一个候选工具名称 100% 在 `rlla_4k/train.parquet`（共 `4,109` 个工具）中出现过**，属于严格意义上的同分布测试集（In-Domain Unseen Queries）。
2. **0% 题目泄露与 100% AST 语法自洽**：与 `train.parquet`（`3,920` 题）及原 `test.parquet`（`80` 题）的用户提问 **0% 重合**；且全部 `500` 道题的 Ground Truth JSON 100% 通过 `parse_tool_calls` 抽象语法树校验（自检 Exact Match = `100.0%`）。
3. **均衡的任务复杂度与数据源分布**：包含 `250` 道单工具调用（`Single-Tool`）与 `250` 道多工具并发调用（`Multi-Tool >= 2`）；同时包含 `250` 道 `xLAM` 标准类型样本与 `250` 道 `ToolACE` 样本。

结合原 `RLLA-4k (80题)`、`BFCL-v3 标准集 (1,240题)` 与 `BFCL-v3 Live (1,351题)`，本报告的所有结论均建立在 **`3,171` 道测试题 × `8` 个模型 = `25,368` 条完整推理评测记录** 之上。

---

## 3. 三阶段训练轨迹与最优 Checkpoint 选择实录

<div align="center">
  <img src="../assets/fig2_training_dynamics_entropy_collapse_zh.png" width="98%" alt="三阶段实验演进：SFT 过拟合拐点、RL 收敛峰值与 SFT+RL 策略熵机理"/>
</div>

### 3.1 第一阶段：SFT 训练与验证 Loss 轨迹 (`outputs/sft/sft-rlla4k-Qwen3-4B-Base-rlla-ep2/trainer_log.jsonl`)

| SFT 训练步数 (Epoch) | 训练集损失 (`train_loss`) | 验证集损失 (`eval_loss`) | 学习率 (`lr`) | 梯度范数 (`grad_norm`) | 模型状态与 Checkpoint 选择 |
| :---: | :---: | :---: | :---: | :---: | :--- |
| `Step 10` (`0.16 Ep`) | `0.7852` | — | `1.97e-5` | `1.05` | 快速适应 `<think>` 与 `<tool_call>` 标签 |
| `Step 30` (`0.48 Ep`) | `0.2143` | — | `1.63e-5` | `0.41` | 模板句 `<think>` 交叉熵迅速归零 |
| **`Step 62` (`1.00 Ep`)** | `0.1510` | **`0.1582` (全局最低)** | `1.05e-5` | `0.29` | **`Model 1a` (`checkpoint-62`，SFT 泛化最优点)** |
| `Step 90` (`1.45 Ep`) | `0.0369` | — | `3.50e-6` | `0.26` | 深度记忆训练集特定参数字符串与引号习惯 |
| **`Step 124` (`2.00 Ep`)** | **`0.0196`** | `0.1700` (`+7.5%` 反弹) | `0.00` | `0.22` | **`Model 1b` (`checkpoint-124`，过拟合训练Loss最低点)** |

### 3.2 第二与第三阶段：四组 GRPO 强化学习验证集奖励轨迹 (`Step 0` $\rightarrow$ `Step 120`)

| GRPO 步数 (Epoch) | `Zero-RL GRPO (v1)` (`Model 3a`, Ep 0–4) | `Zero-RL GRPO (v2)` (`Model 3b`, Ep 4–8) | `Full-SFT + GRPO` (`Model 2a`, 负向对照) | `Cold-Start SFT + GRPO (v2)` (`Model 2c`, 正向对照) |
| :---: | :---: | :---: | :---: | :---: |
| **`Step 0` (`0.0 Ep`)** | `+0.4090` (`EM=43.8%`) | `+2.6594` (`EM=53.8%`) | `+2.9558` (`EM=73.8%`) | `+2.5112` (`Fmt=93.8%`, `EM=50.0%`) |
| **`Step 20` (`0.7 Ep`)** | `+2.0752` (`EM=46.3%`) | `+2.6480` (`EM=55.0%`) | `+2.9502` (`EM=73.8%`) | `+2.6237` (`Fmt=93.8%`, `EM=50.0%`) |
| **`Step 40` (`1.3 Ep`)** | `+2.5015` (`EM=51.3%`) | `+2.6812` (`EM=53.8%`) | `+2.9481` (`EM=72.5%`) | `+2.6941` (`Fmt=95.0%`, `EM=52.5%`) |
| **`Step 60` (`2.0 Ep`)** | `+2.5210` (`EM=52.5%`) | `+2.7105` (`EM=55.0%`) | `+2.9410` (`EM=72.5%`) | `+2.5890` (`Fmt=93.8%`, `EM=51.3%`) |
| **`Step 80` (`2.7 Ep`)** | `+2.5934` (`EM=52.5%`) | `+2.6710` (`EM=55.0%`) | `+2.9395` (`EM=72.5%`) | `+2.6879` (`Fmt=95.0%`, `EM=51.3%`) |
| **`Step 100` (`3.3 Ep`)** | `+2.6218` (`EM=53.8%`) | `+2.7180` (`EM=55.0%`) | `+2.9370` (`EM=72.5%`) | **`+2.7486` (`Fmt=96.3%`, 全局峰值!)** |
| **`Step 120` (`4.0 Ep`)** | **`+2.6584` (`EM=53.8%`, 峰值)** | **`+2.7421` (`EM=56.3%`, 峰值)** | `+2.9370` (`EM=72.5%`) | `+2.7136` (`Fmt=95.0%`, `EM=50.0%`) |

---

## 4. 典型案例逐行对比：为什么 RL 在同分布算术推理与域外多工具组合上完胜 SFT？

### 案例 1：同分布隐式算术偏移量推导 (`rlla_test_20`)
- **用户提问**：*"Fetch 10 marketing email templates starting from the **20th template** (`skip` 为从 0 开始的偏移量，`limit` 为获取数量)。"*
- **Ground Truth**：`{"name": "get_email_templates", "parameters": {"limit": 10, "skip": 19}}`
- **`Pure SFT` (`Model 1a` & `1b`，得分 `0.0`，错误类型 `E6_Param_Value_or_Arithmetic_Error`)**：
  ```xml
  <think> I should use the appropriate tool with proper parameters to respond to the user's need. </think>
  <tool_call>
  {"name": "get_email_templates", "parameters": {"limit": 10, "skip": 20}}
  </tool_call>
  ```
  *失败原因*：由于 `<think>` 被伪模板覆盖，模型在生成 `"skip": ` Token 时没有经过 `20 - 1 = 19` 的中间推理，直接照抄了题干中的字面数字 `20`。
- **`Pure RL` (`Model 3b`) 与 `Cold-Start SFT + GRPO` (`Model 2c`，得分 `1.0`)**：
  ```xml
  <think> The user wants 10 templates starting from the 20th template. Since `skip` is a 0-indexed offset, starting at the 20th item means skipping the first 19 templates (skip = 20 - 1 = 19), and setting limit = 10. </think>
  <tool_call>
  {"name": "get_email_templates", "parameters": {"limit": 10, "skip": 19}}
  </tool_call>
  ```

### 案例 2：同分布闭区间年份跨度计算 (`rlla_test_55`)
- **用户提问**：*"Simulate annual population growth from **Year 70 to Year 80 inclusive**."*
- **Ground Truth**：`{"name": "simulate_growth", "parameters": {"years": 11, ...}}`
- **`Pure SFT` (得分 `0.0`)**：输出伪 `<think>` 后直接填 `"years": 10`（漏加闭区间端点 `+1`）。
- **`Pure RL` 与 `Cold-Start SFT + GRPO` (得分 `1.0`)**：在 `<think>` 中显式写出 *"from Year 70 to Year 80 inclusive covers 80 - 70 + 1 = 11 years"*，准确填入 `"years": 11`。

### 案例 3：分布外多工具选型与并发调用 (`BFCL-v3 parallel_multiple`)
- **任务场景**：在 6 个未见过的物理与数学工具中，挑选 2 个不同工具并发调用并填入完整必填参数。
- **`Pure SFT` (`Model 1b`，得分 `0.0`)**：同时触发 **`E7` 类型引号污染**（将数值写为 `"5.0"`）与 **`E4` 漏填必填参数**（在第二个并发工具调用中漏掉必填字段）。
- **`Cold-Start SFT + GRPO` (`Model 2c`) 与 `Pure RL` (`Model 3b`，得分 `1.0`)**：通过 `<think>` 逐一列出待调用的 2 个工具名及各自的 `required` 字段列表与 JSON 类型，实现零漏参、零引号污染的精确调用。

---

## 5. 工业界落地选型指南：何时用 SFT、何时用 Pure RL、何时用 Cold-Start SFT + GRPO？

| 业务落地场景 | 推荐训练范式 | 实证依据与工程收益 |
| :--- | :--- | :--- |
| **1. 封闭固定工具集场景（内部固定 <20 个 API，极致低延迟）** | **`Pure SFT (1.0 Ep, 验证集 Loss 最低点)`** | 8 卡仅需 `10 分钟` 即可在 `RLLA-80` 达到 `73.75%`（正常子集 `79.69%`），且输出短（~328 字符，无长 CoT 延迟）；务必在 `eval_loss` 最低点（1.0 Ep）早停以避免第 2 个 Epoch 的过度拒答退化。 |
| **2. 开放工具市场 / MCP 生态（动态接入成百上千个第三方未见 API）** | **`Pure RL (Zero-RL GRPO, v2 反作弊奖励)`** | 在未见工具基准 `BFCL-v3 (1k)` 上达到 **`81.50%` 严格 EM（`83.40%` 归一化 EM）**，超越全量 SFT **`+21.10%`**；在规范同分布 `xLAM-250` 上亦达到 **`82.80%`**，100% 保留真实推理链。 |
| **3. 追求最快 RL 收敛与最高真实多候选鲁棒性的企业级 Agent** | **`Cold-Start SFT (320条真CoT, 1 Ep) + GRPO (v2 奖励)`** | 仅用 5 步微缩真 CoT SFT 将初始格式合规率提升至 `93.75%` 同时保留策略熵 `0.492`，随后接入 GRPO 在 `Step 100` 达到最高验证集奖励 `+2.7486`，在真实企业级 `BFCL-Live (1,351题)` 上取得全场最高 **`58.99%` 严格 EM**！ |
