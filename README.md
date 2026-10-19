# agent-rl-lab

基于 **VeRL** 的 Agent 工具调用强化学习（Tool-Use & Agentic RL）实验与方法研究仓库。

## 实验路线图

1. **Stage 1：单轮/多步工具规划与细粒度奖励（ToolRL / RLLA-4k）**
   - **目标**：跑通基于可验证规则奖励（RLVR）的 Function Calling GRPO 训练，掌握格式奖励与细粒度工具参数匹配奖励的设计。
   - **数据集**：`ToolRL rlla_4k`（源自 ToolACE 2,000 条 + xLAM Function Calling 1,000 条 + Hammer 1,000 条；训练集 3,920 条，测试集 80 条）。
   - **奖励机制（`rewards/rlla_reward.py`）**：
     - `format_reward`（`[0.0, 1.0]`）：校验 `<think>...</think>`、`<tool_call>...</tool_call>`、`<response>...</response>` 结构合法性。
     - `accuracy_reward`（`[-3.0, 3.0]`）：细粒度计算工具名称多重集匹配分（`tool_name_match`）+ 参数键匹配分 + 参数值精确匹配分。
     - **结构塌陷监控**：实时输出 `is_collapsed` 与 `exact_match`，用于监控 RL 训练中的控制 token 塌陷现象。

2. **Stage 2：多轮工具调用中的结构塌陷诊断与监督信号（BFCL-V3 / PRS & ETS）**
   - 对比纯 GRPO、SFT+GRPO 与交错监督（Interleaved SFT+RL），复现并缓解控制 token 概率异常放大导致的结构塌陷。

3. **Stage 3：真实环境交互的多轮 Agent Loop（Multi-turn Tool Execution & Search-R1）**
   - 接入代码沙盒（Python Interpreter）与检索工具，进行多轮 `<think> -> <tool_call> -> <obs> -> <think>` 闭环强化学习。

---

## 目录结构

```text
agent-rl-lab/
├── data/
│   └── rlla_4k/              # ToolRL rlla_4k 训练集 (3920) 与测试集 (80)
├── rewards/
│   └── rlla_reward.py        # 细粒度工具调用规则奖励函数 + 结构塌陷诊断指标
├── scripts/
│   ├── download_rlla4k.sh    # 通过 hf-mirror 下载 rlla_4k Parquet 数据集
│   ├── inspect_dataset.py    # 数据集分布统计与 Reward 函数单元测试
│   └── run_grpo_rlla.sh      # 4机32卡 / 单机8卡 VeRL GRPO 训练启动脚本
├── configs/                  # 实验配置存档
├── eval/                     # 离线评测与轨迹分析脚本
└── logs/                     # 训练与评测日志
```

---

## 快速开始（Stage 1：ToolRL RLLA-4k）

### 1. 下载并验证数据集与奖励函数

```bash
# 若尚未下载数据：
bash scripts/download_rlla4k.sh

# 查看数据集统计并运行 Reward 自测：
/home/ma-user/anaconda3/envs/verl_train/bin/python scripts/inspect_dataset.py
```

### 2. 启动 GRPO 训练

默认连接当前 4 机 32 卡 Ray 集群（`172.18.39.173:6379`）：

```bash
# 使用 4 机 32 卡训练 Qwen3-4B-Base
bash scripts/run_grpo_rlla.sh

# 或者先用单机 8 卡快速冒烟验证：
NNODES=1 bash scripts/run_grpo_rlla.sh
```
