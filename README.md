# Agent-RL-Lab: Tool-Use SFT vs. RL (VeRL GRPO + LLaMA-Factory)

基于 **VeRL (GRPO)** 与 **LLaMA-Factory (SFT)** 的工具调用（Function Calling / Tool-Use）对照实验仓库。

- **技术报告全文**：[docs/SFT_vs_RL_Tool_Use_Deep_Dive.md](docs/SFT_vs_RL_Tool_Use_Deep_Dive.md)

---

## 1. 实验设置与评测结果

本实验基于 **Qwen3-4B-Base**，在同一份工具调用训练集 `ToolRL / RLLA-4k`（3,920 条）上训练了 5 组对照模型，并在 3 个评测集（共 2,431 题）上评估模型的域内拟合能力与域外泛化能力：

1. **RLLA-4k Test（80 题，域内测试集 In-Domain）**：与训练集同源同分布（包含 64 道正常样本与 16 道原数据集转换异常样本）。
2. **BFCL-v3 Standard（1,000 题，域外标准集 Out-of-Domain）**：UC Berkeley Function Calling 评测基准（涵盖 `Simple 400`、`Multiple 200`、`Parallel 200`、`Parallel-Multiple 200`），测试集中的工具在训练集中均未出现。
3. **BFCL-v3 Live（1,351 题，域外真实请求集 Out-of-Domain）**：真实开发者提交的线上请求，单题挂载 10~20 个候选工具。

![SFT 与 RL 在三个评测集上的准确率对比](assets/fig1_sft_vs_rl_overview.png)

### 五组对照模型评测结果

| 模型配置 | 训练耗时 | `RLLA-4k` 域内测试集<br>（全集 80题 / 正常样本 64题） | `BFCL-v3` 域外标准集<br>（1,000题 · Strict / Norm EM） | `BFCL-Live` 真实请求集<br>（1,351题 · Strict / Norm EM） | 有效 CoT 占比 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **0. Zero-Shot Base** (`Qwen3-4B-Base-rlla`) | - | `43.75%` / `54.69%` | `60.20%` / `63.60%` | `45.30%` / `51.89%` | `93.8%` |
| **1. Pure SFT** (`3,920条 × 2 Epochs`) | 8 卡 10 分钟 | **`73.75%`** / **`79.69%`** | `60.40%` / `64.60%` | `49.30%` / `58.92%` | `0.0%` |
| **2. Full-SFT + RL** (`2Ep SFT + 120步 GRPO`) | 10 分钟 + 1.5 小时 | `72.50%` / `78.12%` | `59.70%` / `63.90%` | `48.93%` / `58.77%` | `0.0%` |
| **3. Cold-Start SFT** (`320条 × 1 Epoch`) | 8 卡 1 分钟 | `50.00%` / `60.94%` | `67.10%` / `70.50%` | `51.81%` / `59.44%` | `95.0%` |
| **4. Pure RL** (`Zero-RL GRPO 4 Epochs, v1`) | 32 卡 1.5 小时 | `56.25%` / `70.31%` | `79.70%` / `81.90%` | `58.11%` / `64.99%` | `97.5%` |
| **5. Pure RL** (`GRPO 8 Epochs, v2 Reward`) | 32 卡 3.0 小时 | `57.50%` / **`71.88%`** | **`81.50%`** / **`83.40%`** | **`58.77%`** / **`65.58%`** | **`98.8%`** |

---

## 2. 核心实验分析

### 2.1 域内拟合与域外泛化对比
- **域内测试集（`RLLA-4k`）**：在拥有同分布训练数据的前提下，Pure SFT 仅需 10 分钟即可达到 `73.75%`（64 道正常样本上为 `79.69%`），能够快速拟合目标数据集的参数约定与拒答格式。
- **域外测试集（`BFCL-v3` & `BFCL-Live`）**：在未见过的工具集合上，Pure SFT 在 `BFCL-v3` 上仅为 `60.40%`（相对 Base 模型 `60.20%` 提升 `0.2%`）；Pure RL 达到 `81.50%`（高于 Pure SFT `21.1` 个百分点）。在复杂度最高的 `Parallel-Multiple`（多候选+多工具并发调用）子集上，Pure RL 为 `76.50%`，Pure SFT 为 `41.50%`（相差 `35.0` 个百分点）。
- **RLLA-4k 数据质量审计**：`RLLA-4k` 测试集中有 16 道样本（占 20%）存在原数据集转换导致的未闭合 JSON 字符串截断（6 题）或违反 JSON Schema 的类型引号标注（10 题）。在这 16 道异常样本上，Pure SFT 因拟合了训练集中的同类转换缺陷得到 `50.00%`，而 Pure RL 因输出合法 JSON 得到 `0.00%`；在其余 64 道正常样本上，Pure RL 准确率为 `71.88%`。

![RLLA-4k 标注质量拆分与 BFCL-v3 各子集表现对比](assets/fig3_deep_dive_clean_dirty_and_bfcl.png)

### 2.2 Full-SFT + GRPO 奖励停滞原因：模板化 CoT 与策略熵坍缩
- `RLLA-4k` 原始训练集的全部 3,920 条样本中，`<think>` 字段均为固定模板句（`"I should use the appropriate tool with proper parameters to respond to the user's need."`）。
- 全量 2 Epochs SFT 通过 Token 级交叉熵损失强制拟合该模板句后，模型的策略熵（`actor/entropy`）从 `0.72` 降至 `0.08`。
- 后续接续 GRPO 训练时，同一 Prompt 下的 $G=8$ 次 Rollout 采样输出完全一致，导致组内奖励标准差 $\text{std}(r)=0$，优势函数梯度归零（训练 120 步后，验证集 75/80 题输出与 Step 0 完全相同）。
- 相比之下，Pure RL 的奖励函数仅校验 `<tool_call>` 内的 JSON 结果，不约束 `<think>` 内容，模型得以保留并强化自身的真实推理过程。若需在 RL 前引入 SFT 冷启动，应使用包含真实推理链的少量样本（如 `Model 3: Cold-Start SFT`，320 条拒绝采样样本训练 1 Epoch，耗时 1 分钟，保持 `95.0%` 有效推理率）。

![Pure RL 与 Full-SFT + GRPO 的训练过程对比](assets/fig2_training_dynamics_entropy_collapse.png)

### 2.3 细粒度奖励函数设计与 Reward Hacking 抑制（`rewards/rlla_reward_v2.py`）
为提供稠密梯度信号并抑制模型在 RL 训练中的 Reward Hacking 行为，`rlla_reward_v2.py` 在 `[-3.0, +4.0]` 区间内设计了以下机制：
- **冗余参数惩罚（`-0.75 / extra_key`）**：在 `v1` 奖励函数下，模型倾向于输出 Schema 中的所有可选参数以提高命中率。`v2` 对未出现在 Ground Truth 中的多余参数键施加 `-0.75` 分惩罚，消除冗余参数幻觉。
- **非精确匹配封顶（Cap at `+3.4`）**：对存在轻微类型偏差或局部参数错误的输出，总分上限封顶为 `+3.4` 分（精确匹配为 `+4.0` 分），保留 `0.6` 分的优势间隔以驱动模型收敛至严格匹配。

![rlla_reward_v2.py 的分段奖励函数设计](assets/fig4_reward_v2_staircase.png)

### 2.4 Qwen3-Base 特殊 Token 处理与跨框架对齐
- **剥离未初始化特殊 Token（`scripts/prepare_qwen3_base_tokenizer.py`）**：`Qwen3-4B-Base` 词表中的 `151644..151668`（`<|im_end|>`、`<think>`、`<tool_call>` 等）在预训练阶段未被训练，其 Embedding 向量与 2,227 个未使用的生僻字词表项完全相同（`norm=0.3574`）。脚本将其从 `added_tokens` 中移除，使其切分为预训练充分的常规 BPE 子词，并将已训练的 `<|endoftext|>`（`151643`）设为唯一 EOS Token，使 Step 0 验证集准确率从 `0.00%` 恢复至 `43.75%`。
- **LLaMA-Factory 与 VeRL 的 Chat Template 对齐（`configs/sft_qwen3_4b_base_rlla.yaml`）**：在 LLaMA-Factory 配置中省略 `template` 字段，直接调用模型目录下的 `tokenizer.chat_template`（`replace_eos=False`），保证 SFT 与 VeRL 的 Token 序列完全一致。

---

## 3. 目录结构

```text
agent-rl-lab/
├── assets/                                  # 实验结果图表
├── docs/
│   └── SFT_vs_RL_Tool_Use_Deep_Dive.md      # 实验分析报告全文
├── configs/
│   ├── sft_qwen3_4b_base_rlla.yaml          # LLaMA-Factory 全量 SFT 配置
│   └── sft_coldstart_qwen3_4b_base_rlla.yaml# 320 条拒绝采样样本冷启动 SFT 配置
├── rewards/
│   ├── rlla_reward.py                       # ToolRL 原始细粒度规则奖励函数 (v1)
│   └── rlla_reward_v2.py                    # 抑制 Reward Hacking 的改进版奖励函数 (v2)
├── scripts/
│   ├── download_rlla4k.sh                   # 下载 RLLA-4k 数据集
│   ├── prepare_qwen3_base_tokenizer.py      # Qwen3-Base 特殊 Token 剥离与 EOS 修复
│   ├── prepare_sft_data.py                  # Parquet 转 ShareGPT JSON 及奖励自检
│   ├── build_coldstart_sft_data.py          # 拒绝采样筛选 320 条有效 CoT 冷启动数据
│   ├── run_sft_rlla.sh                      # 启动全量 SFT 训练 (8x GPU DeepSpeed ZeRO-2)
│   ├── run_sft_coldstart_rlla.sh            # 启动冷启动 SFT 训练
│   ├── run_grpo_rlla.sh                     # 启动 VeRL GRPO 训练 (v1)
│   ├── run_grpo_rlla_v2.sh                  # 启动 VeRL GRPO 训练 (v2 奖励)
│   └── generate_github_figures.py           # 生成 assets/ 下的图表
└── eval/
    ├── compare_models.py                    # RLLA-4k 域内评测脚本
    ├── run_multi_benchmark_eval.py          # RLLA-4k + BFCL-v3 (1,320题) 联合评测
    ├── run_bfcl_live_eval.py                # BFCL-v3 Live (1,351题) 评测
    ├── summarize_multi_benchmark.py         # 汇总生成评测报告 JSON 与 Markdown
    ├── multi_benchmark_report.md            # 各子集详细评测数据表
    └── rlla4k_model_comparison.json         # RLLA-4k 评测结果 JSON
```

---

## 4. 复现步骤

### 4.1 数据下载与 Tokenizer 处理

```bash
# 下载 RLLA-4k 数据集
bash scripts/download_rlla4k.sh

# 构建处理特殊 Token 后的 Qwen3-4B-Base-rlla 模型目录
python3 scripts/prepare_qwen3_base_tokenizer.py

# 转换 SFT 训练数据并执行 Reward 校验
python3 scripts/prepare_sft_data.py
```

### 4.2 运行 GRPO 强化学习训练

```bash
# 阶段 1：从 Base 模型启动 4 Epochs GRPO (v1 奖励)
bash scripts/run_grpo_rlla.sh

# 阶段 2：使用 v2 奖励函数继续训练
bash scripts/run_grpo_rlla_v2.sh
```

### 4.3 运行 SFT 训练（全量 SFT / 冷启动 SFT）

```bash
# 全量 2 Epochs SFT（单机 8 卡约 10 分钟）
bash scripts/run_sft_rlla.sh

# 拒绝采样 320 条样本 + 1 Epoch 冷启动 SFT
python3 scripts/build_coldstart_sft_data.py
bash scripts/run_sft_coldstart_rlla.sh
```

### 4.4 运行评测

```bash
# 评测 RLLA-4k (80题) + BFCL-v3 标准集 (1,240题)
python3 eval/run_multi_benchmark_eval.py
python3 eval/summarize_multi_benchmark.py

# 评测 BFCL-v3 Live (1,351题)
python3 eval/run_bfcl_live_eval.py
```
