# 工具调用场景下 SFT 与 RL 的效果边界分析：基于 RLLA-4k 与 BFCL-v3 的对照实验报告

[English](SFT_vs_RL_Tool_Use_Deep_Dive.md) | **简体中文**

## 摘要

在构建工具调用（Function Calling / Tool-Use）模型时，工程上常面临几个具体问题：
1. 如果目标是提升某一固定测试集（或固定垂域业务接口）的准确率，直接做 SFT 是否比 RL 更高效？
2. 为什么在全量 SFT 模型基础上继续进行 GRPO 强化学习，奖励曲线往往停滞不前？
3. 在纯垂域场景、无 `<think>`（直接输出 `<tool_call>`）或线上缺少标准答案（Ground Truth）的场景下，RL 的实际作用是什么？

本文基于 **Qwen3-4B-Base**，打通 **LLaMA-Factory (SFT)** 与 **VeRL (GRPO)** 训练流程，在同一份工具调用数据集 `ToolRL / RLLA-4k`（3,920 条）上训练了 5 组对照模型，并在 **RLLA-4k（域内 80 题）**、**BFCL-v3 标准集（域外 1,000 题）** 和 **BFCL-v3 Live（真实用户请求 1,351 题）** 共 2,431 道样本上进行了评测与错误归因分析。

---

## 1. 实验设置与评测结果

所有对比模型均仅使用 `RLLA-4k` 训练集（3,920 条）进行训练，评测涵盖以下三个测试集：

- **RLLA-4k Test（80 题，域内测试集 In-Domain）**：与训练集来自同一数据源，工具定义风格、默认参数习惯及数据转换特征完全一致。
- **BFCL-v3 Standard（1,000 题，域外标准集 Out-of-Domain）**：UC Berkeley Function Calling 评测基准（包含 `Simple 400`、`Multiple 200`、`Parallel 200`、`Parallel-Multiple 200`），测试集中的 API 在训练集中均未出现。
- **BFCL-v3 Live（1,351 题，域外真实请求集 Out-of-Domain）**：来自真实线上用户的工具调用请求，单题通常包含 10~20 个相似候选工具，Prompt 较长。

![In-Domain Memorization (SFT) vs. Out-of-Domain Generalization (RL)](../assets/fig1_sft_vs_rl_overview.png)

| 模型配置 | 训练耗时与算力 | `RLLA-4k` 域内测试集<br>（全集 80题 / 正常样本 64题） | `BFCL-v3` 域外标准集<br>（1,000题 · Strict / Norm EM） | `BFCL-Live` 真实请求集<br>（1,351题 · Strict / Norm EM） | 有效 CoT 占比 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **0. Zero-Shot Base** (`Qwen3-4B-Base-rlla`) | - | `43.75%` / `54.69%` | `60.20%` / `63.60%` | `45.30%` / `51.89%` | `93.8%` |
| **1. Pure SFT** (`3,920条 × 2 Epochs`) | 8 卡 10 分钟 | **`73.75%`** / **`79.69%`** | `60.40%` / `64.60%` | `49.30%` / `58.92%` | `0.0%` |
| **2. Full-SFT + RL** (`2Ep SFT + 120步 GRPO`) | 10 分钟 + 1.5 小时 | `72.50%` / `78.12%` | `59.70%` / `63.90%` | `48.93%` / `58.77%` | `0.0%` |
| **3. Cold-Start SFT** (`320条 × 1 Epoch`) | 8 卡 1 分钟 | `50.00%` / `60.94%` | `67.10%` / `70.50%` | `51.81%` / `59.44%` | `95.0%` |
| **4. Pure RL** (`Zero-RL GRPO 4 Epochs, v1`) | 32 卡 1.5 小时 | `56.25%` / `70.31%` | `79.70%` / `81.90%` | `58.11%` / `64.99%` | `97.5%` |
| **5. Pure RL** (`GRPO 8 Epochs, v2 Reward`) | 32 卡 3.0 小时 | `57.50%` / **`71.88%`** | **`81.50%`** / **`83.40%`** | **`58.77%`** / **`65.58%`** | **`98.8%`** |

从实验数据可以得出以下结论：

1. **针对固定域内测试集（In-Domain），SFT 的训练效率与拟合精度最高**：
   - Pure SFT 在 8 张 A800 上训练 10 分钟，在 `RLLA-4k` 测试集上即达到 `73.75%`（64 道正常样本上为 `79.69%`），耗时仅为 RL 的 1/18。
   - 当训练数据与测试数据同分布时，SFT 的 Token 级监督能直接拟合目标数据集的接口命名、默认参数填充约定及固定拒答句式。
2. **针对未见工具的域外泛化场景（Out-of-Domain），RL 显著优于 SFT**：
   - 在 `BFCL-v3` 的 1,000 道域外测试题上，Pure SFT 准确率为 `60.40%`，相比未训练的 Base 模型（`60.20%`）仅提升 `0.2%`；
   - Pure RL（8 Epochs）准确率达到 `81.50%`（Normalized EM `83.40%`），高出 Pure SFT `21.1` 个百分点；在 `BFCL-Live`（1,351 题）上高出 Pure SFT `9.5` 个百分点。

---

## 2. 域内与域外表现差异分析：RLLA-4k 数据质量审计

为分析 Pure RL 在 `BFCL-v3` 上达到 `81.50%` 而在域内 `RLLA-4k` 全集上仅为 `57.50%` 的原因，我们对 `RLLA-4k` 的 3,920 条训练集和 80 条测试集进行了语法与 Schema 一致性检查。

![In-Domain Label Quality Audit & Out-of-Domain Complexity Scaling](../assets/fig3_deep_dive_clean_dirty_and_bfcl.png)

### 2.1 RLLA-4k 中的标注异常样本
`RLLA-4k` 数据集由 ToolACE、xLAM 和 Hammer 转换而来。原数据集在转换嵌套参数时使用了简单的 `.split(",")` 切分，导致部分 Ground Truth 存在两类缺陷（测试集共 16/80 题，占比 20%）：

1. **未闭合的截断 JSON 字符串（测试集 6 题，训练集 536 题）**：
   例如测试集第 18 题，Ground Truth 中的参数值被截断为不完整的 JSON 字符串：
   ```json
   {"name": "create_student_profile", "parameters": {"courses": "[{\"subject\": \"Mathematics\""}}
   ```
2. **与 JSON Schema 冲突的类型引号标注（测试集 10 题，训练集 945 题）**：
   Prompt 中的工具 Schema 定义为 `integer` 或 `boolean`，但 Ground Truth 将参数值标注为带引号的字符串（如 `"1"`、`"True"`）。

### 2.2 正常样本与异常样本上的表现拆解
- 在 **16 道标注异常样本** 上：Pure RL 的准确率为 `0.00%`（0/16），因为模型生成的是闭合、符合 Schema 类型的合法 JSON，无法与截断的 Ground Truth 严格匹配；而 Pure SFT 在这 16 道题上达到了 `50.00%`（8/16），原因是 SFT 通过 Teacher Forcing 记住了训练集中 536 道同类截断样本的格式特征。
- 在 **64 道正常样本** 上：Pure RL 的准确率上升至 `71.88%`（46/64），与 Pure SFT（`79.69%`，51/64）仅相差 5 道题。

---

## 3. 固定垂域场景下 RL 的作用边界

如果模型仅服务于某一个固定垂域（不涉及 `BFCL` 等域外工具），RL 在垂域内部的增益主要体现在何处？我们对比了 `RLLA-4k` 域内测试集中 **Pure SFT 预测错误、而 Pure RL 预测正确** 的样本。

### 3.1 含 `<think>` 推理链时：解决垂域内的隐式算术与参数换算
在固定垂域中，约 80% 的请求属于直接槽位填充（SFT 可很好覆盖），但仍有 15%~20% 的请求需要经过一步算术或逻辑换算才能得到正确的参数值。

**案例 1：分页偏移量换算（`rlla_test_20`）**
- **用户请求**：查询目录中第 20 个到第 30 个商品（工具 `get_products` 接收 `skip` 和 `limit`）。
- **Pure SFT 输出（错误）**：直接提取文本中的数字 `20` 填入 `{"limit": 10, "skip": 20}`（跳过前 20 个实际返回的是第 21~30 个商品）。
- **Pure RL 输出（正确）**：在 `<think>` 中完成了 `20 - 1 = 19` 的偏移计算，输出 `{"skip": 19, "limit": 10}`。

**案例 2：闭区间计数换算（`rlla_test_55`）**
- **用户请求**：拉取下标 70 到 80（包含两端）的行情数据。
- **Pure SFT 输出（错误）**：直接填入数量 `10`。
- **Pure RL 输出（正确）**：在 `<think>` 中推导出包含两端共需拉取 `80 - 70 + 1 = 11` 条记录，输出 `11`。

### 3.2 若去掉 `<think>`（直接输出 `<tool_call>`）时 RL 的作用
如果业务为降低在线推理延迟而完全移除 `<think>`，模型失去显式推理步骤，RL 在参数算术推导上的优势会大幅减弱。在此设定下，垂域优化的重点应放在高质量 SFT 数据构建上，RL 仅在以下三个方面提供补充：
1. **序列级奖惩约束（消除局部漏参或冗余参数）**：SFT 采用 Token 级交叉熵损失，一段 50 Token 的输出中若仅漏掉最后一个工具的参数或多出一个可选参数，其 Token Loss 依然很低；而 RL 按整个工具调用序列打分，可直接对局部错误施加惩罚（例如 `rlla_test_34` 连续调用 3 个工具时，Pure SFT 在第 3 个工具处漏填参数输出 `{}`，Pure RL 完整填入了全部参数）。
2. **调整工具调用与拒答反问的阈值**：通过调整拒答奖励与错误调用惩罚的权重比，控制模型在缺参场景下的保守程度。
3. **利用无标注的线上真实 Query 进行迭代**：结合规则校验器在无 Ground Truth 的线上请求上进行强化学习。

---

## 4. Full-SFT + GRPO 奖励停滞原因分析：模板化 CoT 与策略熵坍缩

在 `Model 2 (Full-SFT + RL)` 实验中，先在 3,920 条训练集上进行 2 Epochs 全参 SFT，随后接续 150 步 GRPO 训练，验证集平均奖励停留在 `+2.9558 -> +2.9370`。

![Training Dynamics: Pure RL vs. Full-SFT + GRPO](../assets/fig2_training_dynamics_entropy_collapse.png)

1. **SFT 与 RL 对 `<think>` 处理机制的差异**：`RLLA-4k` 原始训练集的全部 3,920 条样本中，标准答案的 `<think>` 字段均为同一句固定模板。SFT 将模板句作为 Token 级监督目标，训练 2 Epochs 后模型策略熵（`actor/entropy`）从 `0.72` 降至 `0.08`，有效 CoT 占比降至 `0.0%`；而 Pure RL 的奖励函数仅校验 `<tool_call>` 内的 JSON，忽略标准答案中的模板 `<think>`，使模型保留了自身的真实推理链（`98.8%`）。
2. **策略熵坍缩导致 GRPO 优势梯度归零**：在策略熵降至 `0.08` 后，同一 Prompt 下的 $G=8$ 次采样输出完全一致，组内奖励标准差 $\text{std}(r) = 0$，导致优势函数 $A_i = 0$、梯度归零（训练 120 步后 75/80 题输出与 Step 0 完全相同）。

---

## 5. 细粒度奖励函数设计与无 GT 场景扩展

![Dense Reward Shaping & Anti-Reward-Hacking Design](../assets/fig4_reward_v2_staircase.png)

### 5.1 抑制 Reward Hacking（`rewards/rlla_reward_v2.py`）
1. **冗余参数惩罚（Extra-Key Penalty, `-0.75 / key`）**：对预测结果中多出的非 Ground Truth 参数键每个扣减 `0.75` 分，消除模型通过填满所有可选参数博取命中率的行为。
2. **非精确匹配封顶（Non-Exact Match Cap, `+3.4` vs. `+4.0`）**：将未达到 100% 严格匹配的工具调用总分封顶为 `+3.4` 分，与完全匹配（`+4.0` 分）保持 `0.60` 分差值。

### 5.2 生产环境中无 Ground Truth（No-GT）时的奖励构建
在缺少人工标注 Ground Truth 的线上增量训练场景下，可通过三层验证信号构建奖励函数：
1. **Schema 静态校验与沙箱执行**：对照工具 JSON Schema 校验必填项、参数类型、枚举值及沙箱 Dry-run 状态码（零标注成本过滤语法与类型错误）。
2. **执行结果验证（Outcome Check）**：根据查询结果非空或后置断言判别调用有效性。
3. **语义一致性判别（LLM-as-a-Judge）**：离线训练时使用大尺寸模型判别预测参数是否符合用户意图。
