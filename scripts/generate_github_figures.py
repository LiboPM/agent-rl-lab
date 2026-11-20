#!/usr/bin/env python3
"""
Generate high-resolution (300 DPI) analytical figures for GitHub README and technical article.
Saves PNG files to `assets/`.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_dir = PROJECT_ROOT / "assets"
ASSETS_dir.mkdir(parents=True, exist_ok=True)

# Load Chinese font SimHei alongside DejaVu Sans fallback
SIMHEI_PATH = Path("/home/ma-user/work/LB/gitee/omni_agent/docs/design/figures/fonts/SimHei.ttf")
if SIMHEI_PATH.exists():
    fm.fontManager.addfont(str(SIMHEI_PATH))
    plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
else:
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300

# Clean technical palette
C_BASE = "#94a3b8"      # Slate 400
C_SFT = "#3b82f6"       # Blue 500
C_SFT_DARK = "#1d4ed8"  # Blue 700
C_RL = "#10b981"        # Emerald 500
C_RL_DARK = "#047857"   # Emerald 700
C_WARN = "#f59e0b"      # Amber 500
C_DANGER = "#ef4444"    # Red 500
C_PURPLE = "#8b5cf6"    # Violet 500
C_TEXT = "#0f172a"      # Slate 900
C_SUB = "#475569"       # Slate 600


def style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor("#ffffff")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.grid(axis="y", linestyle="--", alpha=0.45, color="#cbd5e1", zorder=0)
    ax.tick_params(colors=C_TEXT, labelsize=10.5)


def fig1_sft_vs_rl_overview() -> Path:
    """Figure 1: 3-Benchmark Comparison (In-Domain RLLA vs OOD BFCL-v3 vs OOD BFCL-Live)."""
    fig, ax = plt.subplots(figsize=(11.5, 6.0), facecolor="#ffffff")
    style_ax(ax)

    benchmarks = [
        "RLLA-4k 测试集\n（80题 · 域内测试集）",
        "BFCL-v3 标准集\n（1,000题 · 域外标准集）",
        "BFCL-Live 真实集\n（1,351题 · 域外真实用户请求）",
    ]
    base_scores = [43.75, 60.20, 45.30]
    sft_scores = [73.75, 60.40, 49.30]
    rl_scores = [57.50, 81.50, 58.77]

    x = np.arange(len(benchmarks))
    width = 0.24

    b0 = ax.bar(x - width, base_scores, width, label="Zero-Shot Base (未训练基座)", color=C_BASE, edgecolor="none", zorder=3, alpha=0.85)
    b1 = ax.bar(x, sft_scores, width, label="纯 SFT (2 Epochs · 8卡 10分钟)", color=C_SFT, edgecolor="none", zorder=3)
    b2 = ax.bar(x + width, rl_scores, width, label="纯 RL (GRPO 8 Epochs · v2奖励)", color=C_RL, edgecolor="none", zorder=3)

    for bars, color in [(b0, C_SUB), (b1, C_SFT_DARK), (b2, C_RL_DARK)]:
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 1.3,
                f"{h:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10.5,
                fontweight="bold",
                color=color,
            )

    ax.annotate(
        "域内表现：纯 SFT 领先 (+16.3%)\n训练耗时仅 10 分钟\n（剔除 16 道异常标注样本后 RL 为 71.9%）",
        xy=(0, 73.75),
        xytext=(-0.36, 86.5),
        fontsize=9.5,
        color=C_SFT_DARK,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#eff6ff", edgecolor="#93c5fd", lw=1.2),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.15", color=C_SFT, lw=1.5),
    )

    ax.annotate(
        "域外表现：纯 RL 领先 +21.1%\nSFT 相比基座几乎无提升 (60.2% → 60.4%)\nRL 保持了未见工具的泛化能力 (81.5%)",
        xy=(1 + width, 81.50),
        xytext=(0.75, 90.0),
        fontsize=9.5,
        color=C_RL_DARK,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#ecfdf5", edgecolor="#6ee7b7", lw=1.2),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.15", color=C_RL, lw=1.5),
    )

    ax.annotate(
        "多候选工具场景：纯 RL 领先 +9.5%\n(单题含 10~20 个候选工具)",
        xy=(2 + width, 58.77),
        xytext=(1.68, 73.0),
        fontsize=9.5,
        color=C_RL_DARK,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#ecfdf5", edgecolor="#6ee7b7", lw=1.2),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.15", color=C_RL, lw=1.5),
    )

    ax.set_ylabel("Strict Exact Match (%)", fontsize=11.5, color=C_TEXT, fontweight="bold")
    ax.set_title(
        "SFT 与 RL 在三个评测集（共 2,431 题）上的准确率对比",
        fontsize=13.5,
        fontweight="bold",
        color=C_TEXT,
        pad=16,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1", fontsize=10.5)

    fig.tight_layout()
    out = ASSETS_dir / "fig1_sft_vs_rl_overview.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def fig2_training_dynamics_entropy_collapse() -> Path:
    """Figure 2: Training Curves from JSONL logs showing Full-SFT + RL vs Pure RL."""
    metrics_dir = PROJECT_ROOT / "logs" / "metrics"
    purerl_log = metrics_dir / "grpo-rlla4k-Qwen3-4B-Base-rlla-n4g8_20261019_190110.jsonl"
    sftrl_log = metrics_dir / "grpo-rlla4k-Qwen3-4B-SFT-ep2-n4g8_20261020_104244.jsonl"

    def parse_log(path: Path):
        val_steps, val_rewards = [], []
        tr_steps, entropies = [], []
        for line in open(path):
            row = json.loads(line)
            step = row.get("step", 0)
            data = row.get("data", {})
            if "val-core/rlla/reward/mean@1" in data:
                val_steps.append(step)
                val_rewards.append(data["val-core/rlla/reward/mean@1"])
            if "actor/entropy" in data:
                tr_steps.append(step)
                entropies.append(data["actor/entropy"])
        return val_steps, val_rewards, tr_steps, entropies

    p_vs, p_vr, p_ts, p_ent = parse_log(purerl_log)
    s_vs, s_vr, s_ts, s_ent = parse_log(sftrl_log)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.5), facecolor="#ffffff")
    style_ax(ax1)
    style_ax(ax2)

    # Panel A: Validation Reward Curve
    ax1.plot(s_vs, s_vr, marker="s", markersize=5, lw=2.4, color=C_DANGER, label="Full-SFT + GRPO (+2.96 → +2.94)")
    ax1.plot(p_vs, p_vr, marker="o", markersize=5, lw=2.4, color=C_RL, label="Pure RL (+0.96 → +2.61)")
    ax1.axhline(2.676, color=C_RL_DARK, linestyle=":", lw=1.8, label="Pure RL + v2奖励 (Epoch 8: +2.68)")

    ax1.annotate(
        "Full-SFT + RL 奖励无明显变化\nStep 120 与 Step 0 相比，75/80 题输出完全一致",
        xy=(80, 2.94),
        xytext=(32, 2.15),
        fontsize=9.5,
        color="#b91c1c",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fef2f2", edgecolor="#fca5a5"),
        arrowprops=dict(arrowstyle="->", color=C_DANGER, lw=1.5),
    )
    ax1.annotate(
        "Pure RL 持续上升 (+1.72)\n97.5% 样本包含有效推理过程",
        xy=(120, 2.61),
        xytext=(55, 1.35),
        fontsize=9.5,
        color=C_RL_DARK,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#ecfdf5", edgecolor="#6ee7b7"),
        arrowprops=dict(arrowstyle="->", color=C_RL, lw=1.5),
    )

    ax1.set_title("(a) 验证集平均奖励 (val-core/rlla/reward/mean@1)", fontsize=12, fontweight="bold", pad=12)
    ax1.set_xlabel("GRPO 训练步数 (Global Step)", fontsize=11)
    ax1.set_ylabel("Validation Mean Reward [-3.0, +4.0]", fontsize=11)
    ax1.set_ylim(0.7, 3.35)
    ax1.legend(loc="lower right", fontsize=9.5, frameon=True, facecolor="#f8fafc")

    # Panel B: Policy Entropy Curve
    ax2.plot(p_ts, p_ent, lw=2.0, color=C_RL, alpha=0.9, label="Pure RL 策略熵 (0.72 → 0.25)")
    ax2.plot(s_ts, s_ent, lw=2.2, color=C_DANGER, alpha=0.9, label="Full-SFT + GRPO 策略熵 (~0.08)")

    ax2.annotate(
        "原因：SFT 训练集 <think> 为固定模板句\n2 Epochs 训练后策略熵降至 0.08\n导致 GRPO 组内 8 次采样结果相同，std(r)=0，梯度为 0",
        xy=(30, 0.085),
        xytext=(20, 0.36),
        fontsize=9.5,
        color="#b91c1c",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#fef2f2", edgecolor="#fca5a5"),
        arrowprops=dict(arrowstyle="->", color=C_DANGER, lw=1.5),
    )

    ax2.set_title("(b) 策略熵变化曲线 (actor/entropy)", fontsize=12, fontweight="bold", pad=12)
    ax2.set_xlabel("GRPO 训练步数 (Global Step)", fontsize=11)
    ax2.set_ylabel("Token-Level Policy Entropy", fontsize=11)
    ax2.set_ylim(0.0, 0.82)
    ax2.legend(loc="upper right", fontsize=9.5, frameon=True, facecolor="#f8fafc")

    fig.suptitle(
        "Pure RL 与 Full-SFT + GRPO 的训练过程对比（奖励与策略熵）",
        fontsize=13.5,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = ASSETS_dir / "fig2_training_dynamics_entropy_collapse.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def fig3_deep_dive_clean_dirty_and_bfcl() -> Path:
    """Figure 3: RLLA Data Quality Breakdown (64 Clean vs 16 Dirty) & BFCL 4 Subcategories."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.5), facecolor="#ffffff")
    style_ax(ax1)
    style_ax(ax2)

    # Left: RLLA-4k Clean vs Dirty Breakdown
    cats_l = [
        "全部测试样本\n(80题)",
        "正常样本\n(64题 · 80%)",
        "标注异常样本\n(16题 · 20%)",
    ]
    sft_l = [73.75, 79.69, 50.00]
    rl_l = [57.50, 71.88, 0.00]

    x1 = np.arange(len(cats_l))
    w1 = 0.30
    b_s1 = ax1.bar(x1 - w1 / 2, sft_l, w1, label="纯 SFT (2 Epochs)", color=C_SFT, zorder=3)
    b_r1 = ax1.bar(x1 + w1 / 2, rl_l, w1, label="纯 RL (8 Epochs)", color=C_RL, zorder=3)

    for bars, c in [(b_s1, C_SFT_DARK), (b_r1, C_RL_DARK)]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold", color=c)

    ax1.annotate(
        "16 道异常样本包含未闭合 JSON 字符串\n（如 \"[{\\\"subject\\\": \\\"Math\\\"）或类型引号不一致\nSFT 拟合了训练集中的同类截断格式 (50%)\nRL 输出合法 JSON 因此无法匹配异常标签 (0%)",
        xy=(2 + w1 / 2, 2.0),
        xytext=(0.88, 30.0),
        fontsize=9.0,
        color="#b45309",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fffbeb", edgecolor="#fcd34d"),
        arrowprops=dict(arrowstyle="->", color=C_WARN, lw=1.5),
    )

    ax1.set_title("(a) RLLA-4k 测试集拆分：64道正常样本 vs 16道异常样本", fontsize=12, fontweight="bold", pad=12)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(cats_l, fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("Strict Exact Match (%)", fontsize=11)
    ax1.set_ylim(0, 98)
    ax1.legend(loc="upper right", fontsize=10)

    # Right: BFCL-v3 4 Subcategories
    cats_r = [
        "Simple\n单工具 (400题)",
        "Multiple\n多选一 (200题)",
        "Parallel\n单工具并发 (200题)",
        "Parallel-Multiple\n多选+并发 (200题)",
    ]
    sft_r = [69.25, 70.50, 51.50, 41.50]
    rl_r = [82.75, 81.50, 84.00, 76.50]

    x2 = np.arange(len(cats_r))
    b_s2 = ax2.bar(x2 - w1 / 2, sft_r, w1, label="纯 SFT (2 Epochs)", color=C_SFT, zorder=3)
    b_r2 = ax2.bar(x2 + w1 / 2, rl_r, w1, label="纯 RL (8 Epochs)", color=C_RL, zorder=3)

    for bars, c in [(b_s2, C_SFT_DARK), (b_r2, C_RL_DARK)]:
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=9.8, fontweight="bold", color=c)

    ax2.annotate(
        "在多工具并发调用场景下差距拉大：\nParallel-Multiple 子集上\n纯 RL (76.5%) 高于纯 SFT (41.5%) 35.0 个百分点",
        xy=(3 + w1 / 2, 76.5),
        xytext=(1.25, 88.0),
        fontsize=9.2,
        color=C_RL_DARK,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#ecfdf5", edgecolor="#6ee7b7"),
        arrowprops=dict(arrowstyle="->", color=C_RL, lw=1.5),
    )

    ax2.set_title("(b) BFCL-v3 域外标准集：四个子任务准确率对比", fontsize=12, fontweight="bold", pad=12)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(cats_r, fontsize=10, fontweight="bold")
    ax2.set_ylabel("Strict Exact Match (%)", fontsize=11)
    ax2.set_ylim(0, 105)
    ax2.legend(loc="upper left", fontsize=10)

    fig.suptitle(
        "RLLA-4k 标注质量拆分与 BFCL-v3 各子集表现对比",
        fontsize=13.5,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = ASSETS_dir / "fig3_deep_dive_clean_dirty_and_bfcl.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def fig4_reward_v2_staircase() -> Path:
    """Figure 4: Stepwise Scoring Structure of `rewards/rlla_reward_v2.py`."""
    fig, ax = plt.subplots(figsize=(11.5, 5.2), facecolor="#ffffff")
    style_ax(ax)

    stages = [
        "阶段 0\nJSON解析失败\n或工具误调",
        "阶段 1\nXML标签合规\n<think>+<tool_call>",
        "阶段 2\n工具名正确\n+ 必填参数Key",
        "阶段 3\n参数值匹配\n(含冗余可选参数)",
        "阶段 4\n非全对分数上限\n(存在类型/局部偏差)",
        "阶段 5\n严格精确匹配\n(Exact Match = 1.0)",
    ]
    scores = [-3.0, 1.0, 2.1, 2.38, 3.4, 4.0]
    colors = [C_DANGER, C_BASE, "#60a5fa", C_WARN, C_PURPLE, C_RL]

    x = np.arange(len(stages))
    bars = ax.bar(x, scores, width=0.52, color=colors, edgecolor="none", zorder=3)
    ax.axhline(0, color="#64748b", lw=1.2, zorder=2)

    for bar, s, c in zip(bars, scores, colors):
        va = "bottom" if s >= 0 else "top"
        offset = 0.15 if s >= 0 else -0.42
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            s + offset,
            f"{s:+.2f} 分",
            ha="center",
            va=va,
            fontsize=11,
            fontweight="bold",
            color=c if s < 0 else C_TEXT,
        )

    ax.annotate(
        "v2 改进 1（冗余参数惩罚）：\n每多输出一个未出现在 GT 中的参数 Key\n扣减 0.75 分",
        xy=(3, 2.38),
        xytext=(2.05, 3.55),
        fontsize=9.5,
        color="#b45309",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fffbeb", edgecolor="#fcd34d"),
        arrowprops=dict(arrowstyle="->", color=C_WARN, lw=1.5),
    )

    ax.annotate(
        "v2 改进 2（非全对封顶）：\n未完全匹配时最高封顶 +3.4 分\n与完全匹配 (+4.0 分) 保持 0.60 分差值",
        xy=(5, 4.0),
        xytext=(3.55, 4.35),
        fontsize=9.5,
        color=C_RL_DARK,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#ecfdf5", edgecolor="#6ee7b7"),
        arrowprops=dict(arrowstyle="->", color=C_RL, lw=1.5),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=10, fontweight="bold")
    ax.set_ylabel("Total Reward Score [-3.0, +4.0]", fontsize=11, fontweight="bold")
    ax.set_ylim(-3.8, 5.1)
    ax.set_title(
        "rlla_reward_v2.py 的分段奖励函数设计",
        fontsize=13.5,
        fontweight="bold",
        pad=15,
    )

    fig.tight_layout()
    out = ASSETS_dir / "fig4_reward_v2_staircase.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


if __name__ == "__main__":
    p1 = fig1_sft_vs_rl_overview()
    p2 = fig2_training_dynamics_entropy_collapse()
    p3 = fig3_deep_dive_clean_dirty_and_bfcl()
    p4 = fig4_reward_v2_staircase()
    print("Generated figures:")
    for p in (p1, p2, p3, p4):
        print(" -", p)
