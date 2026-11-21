#!/usr/bin/env python3
"""
Generate publication-grade (300 DPI) bilingual figures (English `fig*.png` & Chinese `fig*_zh.png`)
for GitHub README (`README.md` / `README_zh.md`) and technical reports.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

SIMHEI_PATH = Path("/home/ma-user/work/LB/gitee/omni_agent/docs/design/figures/fonts/SimHei.ttf")
if SIMHEI_PATH.exists():
    fm.fontManager.addfont(str(SIMHEI_PATH))

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300

# Clean academic color palette
C_BASE = "#94a3b8"      # Slate 400
C_SFT = "#2563eb"       # Blue 600
C_SFT_DARK = "#1e40af"  # Blue 800
C_RL = "#059669"        # Emerald 600
C_RL_DARK = "#065f46"   # Emerald 800
C_WARN = "#d97706"      # Amber 600
C_DANGER = "#dc2626"    # Red 600
C_PURPLE = "#7c3aed"    # Violet 600
C_TEXT = "#0f172a"      # Slate 900
C_SUB = "#475569"       # Slate 600


def set_lang_font(lang: str) -> None:
    if lang == "zh" and SIMHEI_PATH.exists():
        plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]


def style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor("#ffffff")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.grid(axis="y", linestyle="--", alpha=0.45, color="#cbd5e1", zorder=0)
    ax.tick_params(colors=C_TEXT, labelsize=10)


def fig1_sft_vs_rl_overview(lang: str = "en") -> Path:
    set_lang_font(lang)
    fig, ax = plt.subplots(figsize=(11.5, 5.8), facecolor="#ffffff")
    style_ax(ax)

    if lang == "en":
        benchmarks = [
            "RLLA-4k Test (N=80)\n[In-Domain Split]",
            "BFCL-v3 Standard (N=1,000)\n[Out-of-Domain Benchmark]",
            "BFCL-v3 Live (N=1,351)\n[Real-World User Queries]",
        ]
        lbl_base = "Zero-Shot Base (Qwen3-4B-Base-rlla)"
        lbl_sft = "Pure SFT (2 Epochs · 8 GPUs, 10m)"
        lbl_rl = "Pure RL (GRPO 8 Epochs · v2 Reward)"
        note1 = "In-Domain Fit: Pure SFT leads (+16.3%)\nin 10 mins (18x faster than RL)\n*On 64 clean samples: RL reaches 71.9%"
        note2 = "OOD Generalization: Pure RL leads by +21.1%\nSFT stagnates near Base (60.2% -> 60.4%)\nRL generalizes to unseen schemas (81.5%)"
        note3 = "Real-World Robustness: Pure RL +9.5%\n(10-20 candidate tools per prompt)"
        ylabel = "Strict Exact Match (%)"
        title = "In-Domain Memorization (SFT) vs. Out-of-Domain Generalization (RL) Across 2,431 Samples"
        suffix = ""
    else:
        benchmarks = [
            "RLLA-4k 测试集 (N=80)\n[域内同分布测试集]",
            "BFCL-v3 标准集 (N=1,000)\n[域外标准评测集]",
            "BFCL-v3 Live (N=1,351)\n[域外真实用户请求]",
        ]
        lbl_base = "Zero-Shot Base (未训练基座)"
        lbl_sft = "Pure SFT (2 Epochs · 8卡 10分钟)"
        lbl_rl = "Pure RL (GRPO 8 Epochs · v2奖励)"
        note1 = "域内表现：Pure SFT 领先 (+16.3%)\n训练耗时仅 10 分钟\n（64道正常样本上 RL 准确率为 71.9%）"
        note2 = "域外泛化：Pure RL 领先 +21.1%\nSFT 相比基座几乎无提升 (60.2% -> 60.4%)\nRL 保持了未见工具的泛化能力 (81.5%)"
        note3 = "多候选工具场景：Pure RL 领先 +9.5%\n(单题包含 10~20 个候选工具)"
        ylabel = "Strict Exact Match (%)"
        title = "SFT 与 RL 在三个评测集（共 2,431 题）上的准确率对比"
        suffix = "_zh"

    base_scores = [43.75, 60.20, 45.30]
    sft_scores = [73.75, 60.40, 49.30]
    rl_scores = [57.50, 81.50, 58.77]

    x = np.arange(len(benchmarks))
    width = 0.24

    b0 = ax.bar(x - width, base_scores, width, label=lbl_base, color=C_BASE, edgecolor="none", zorder=3, alpha=0.85)
    b1 = ax.bar(x, sft_scores, width, label=lbl_sft, color=C_SFT, edgecolor="none", zorder=3)
    b2 = ax.bar(x + width, rl_scores, width, label=lbl_rl, color=C_RL, edgecolor="none", zorder=3)

    for bars, color in [(b0, C_SUB), (b1, C_SFT_DARK), (b2, C_RL_DARK)]:
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 1.2,
                f"{h:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color=color,
            )

    ax.annotate(
        note1,
        xy=(0, 73.75),
        xytext=(-0.38, 86.5),
        fontsize=9.2,
        color=C_SFT_DARK,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#eff6ff", edgecolor="#93c5fd", lw=1.2),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.15", color=C_SFT, lw=1.4),
    )

    ax.annotate(
        note2,
        xy=(1 + width, 81.50),
        xytext=(0.70, 89.5),
        fontsize=9.2,
        color=C_RL_DARK,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#ecfdf5", edgecolor="#6ee7b7", lw=1.2),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.15", color=C_RL, lw=1.4),
    )

    ax.annotate(
        note3,
        xy=(2 + width, 58.77),
        xytext=(1.60, 72.5),
        fontsize=9.2,
        color=C_RL_DARK,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#ecfdf5", edgecolor="#6ee7b7", lw=1.2),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.15", color=C_RL, lw=1.4),
    )

    ax.set_ylabel(ylabel, fontsize=11, color=C_TEXT, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", color=C_TEXT, pad=16)
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, fontsize=10.5, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1", fontsize=9.8)

    fig.tight_layout()
    out = ASSETS_DIR / f"fig1_sft_vs_rl_overview{suffix}.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def fig2_training_dynamics_entropy_collapse(lang: str = "en") -> Path:
    set_lang_font(lang)
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4), facecolor="#ffffff")
    style_ax(ax1)
    style_ax(ax2)

    if lang == "en":
        l_sft = "Full-SFT + GRPO (+2.96 -> +2.94, Stalled)"
        l_rl = "Pure RL (+0.96 -> +2.61, 4 Epochs)"
        l_v2 = "Pure RL + v2 Reward (Epoch 8: +2.68)"
        n_a1 = "Reward Stagnation in Full-SFT + GRPO:\n75/80 validation outputs at Step 120 are\ncharacter-for-character identical to Step 0"
        n_a2 = "Pure RL climbs steadily (+1.72)\n97.5% samples preserve genuine CoT"
        t_a = "(a) Validation Reward Progression (val-core/rlla/reward/mean@1)"
        l_ent_rl = "Pure RL Policy Entropy (0.72 -> 0.25)"
        l_ent_sft = "Full-SFT + GRPO Policy Entropy (~0.08)"
        n_b = "Root Cause — Policy Entropy Collapse (0.08):\nBoilerplate <think> in SFT data overwrites CoT;\nall G=8 rollouts become identical -> std(r)=0 -> zero gradient"
        t_b = "(b) Token-Level Policy Entropy (actor/entropy)"
        suptitle = "Training Dynamics: Why Full-SFT + GRPO Stalls Due to Boilerplate CoT & Entropy Collapse"
        suffix = ""
    else:
        l_sft = "Full-SFT + GRPO (+2.96 -> +2.94, 停滞)"
        l_rl = "Pure RL (+0.96 -> +2.61, 4 Epochs)"
        l_v2 = "Pure RL + v2奖励 (Epoch 8: +2.68)"
        n_a1 = "Full-SFT + GRPO 奖励停滞：\nStep 120 与 Step 0 相比，\n验证集 75/80 题输出完全一致"
        n_a2 = "Pure RL 持续上升 (+1.72)\n97.5% 样本保留真实推理链"
        t_a = "(a) 验证集平均奖励 (val-core/rlla/reward/mean@1)"
        l_ent_rl = "Pure RL 策略熵 (0.72 -> 0.25)"
        l_ent_sft = "Full-SFT + GRPO 策略熵 (~0.08)"
        n_b = "原因：SFT 数据中 <think> 为固定模板句\n2 Epochs 训练后策略熵降至 0.08\n组内 G=8 次采样输出完全相同 -> std(r)=0 -> 梯度归零"
        t_b = "(b) 策略熵变化曲线 (actor/entropy)"
        suptitle = "Pure RL 与 Full-SFT + GRPO 的训练过程对比（奖励与策略熵）"
        suffix = "_zh"

    ax1.plot(s_vs, s_vr, marker="s", markersize=4.5, lw=2.2, color=C_DANGER, label=l_sft)
    ax1.plot(p_vs, p_vr, marker="o", markersize=4.5, lw=2.2, color=C_RL, label=l_rl)
    ax1.axhline(2.676, color=C_RL_DARK, linestyle=":", lw=1.8, label=l_v2)

    ax1.annotate(
        n_a1,
        xy=(80, 2.94),
        xytext=(30, 2.12),
        fontsize=9.2,
        color="#991b1b",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fef2f2", edgecolor="#fca5a5"),
        arrowprops=dict(arrowstyle="->", color=C_DANGER, lw=1.4),
    )
    ax1.annotate(
        n_a2,
        xy=(120, 2.61),
        xytext=(52, 1.32),
        fontsize=9.2,
        color=C_RL_DARK,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#ecfdf5", edgecolor="#6ee7b7"),
        arrowprops=dict(arrowstyle="->", color=C_RL, lw=1.4),
    )

    ax1.set_title(t_a, fontsize=11.5, fontweight="bold", pad=12)
    ax1.set_xlabel("GRPO Training Steps (Global Step)", fontsize=10.5)
    ax1.set_ylabel("Validation Mean Reward [-3.0, +4.0]", fontsize=10.5)
    ax1.set_ylim(0.7, 3.35)
    ax1.legend(loc="lower right", fontsize=9.2, frameon=True, facecolor="#f8fafc")

    ax2.plot(p_ts, p_ent, lw=2.0, color=C_RL, alpha=0.9, label=l_ent_rl)
    ax2.plot(s_ts, s_ent, lw=2.2, color=C_DANGER, alpha=0.9, label=l_ent_sft)

    ax2.annotate(
        n_b,
        xy=(30, 0.085),
        xytext=(16, 0.36),
        fontsize=9.2,
        color="#991b1b",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#fef2f2", edgecolor="#fca5a5"),
        arrowprops=dict(arrowstyle="->", color=C_DANGER, lw=1.4),
    )

    ax2.set_title(t_b, fontsize=11.5, fontweight="bold", pad=12)
    ax2.set_xlabel("GRPO Training Steps (Global Step)", fontsize=10.5)
    ax2.set_ylabel("Policy Entropy", fontsize=10.5)
    ax2.set_ylim(0.0, 0.82)
    ax2.legend(loc="upper right", fontsize=9.2, frameon=True, facecolor="#f8fafc")

    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = ASSETS_DIR / f"fig2_training_dynamics_entropy_collapse{suffix}.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def fig3_deep_dive_clean_dirty_and_bfcl(lang: str = "en") -> Path:
    set_lang_font(lang)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4), facecolor="#ffffff")
    style_ax(ax1)
    style_ax(ax2)

    if lang == "en":
        cats_l = ["All Test Samples\n(N=80)", "Clean Samples\n(N=64 · 80%)", "Corrupted GT\n(N=16 · 20%)"]
        n_l = "16 Corrupted GTs have truncated unclosed JSON\n(e.g. \"[{\\\"subject\\\": \\\"Math\\\") or quoted ints.\nSFT memorizes the dataset truncation bug (50%);\nRL emits valid JSON and scores 0% on broken GT."
        t_l = "(a) RLLA-4k Audit: 64 Clean vs. 16 Corrupted GTs"
        cats_r = ["Simple\n(N=400)", "Multiple\n(N=200)", "Parallel\n(N=200)", "Parallel-Multiple\n(N=200)"]
        n_r = "Advantage scales with task complexity:\nOn Parallel-Multiple, Pure RL (76.5%)\noutperforms Pure SFT (41.5%) by +35.0%"
        t_r = "(b) BFCL-v3 Out-of-Domain Breakdown by Difficulty"
        suptitle = "In-Domain Label Quality Audit (RLLA-4k) & Out-of-Domain Complexity Scaling (BFCL-v3)"
        suffix = ""
    else:
        cats_l = ["全部测试样本\n(N=80)", "正常样本\n(N=64 · 80%)", "标注异常样本\n(N=16 · 20%)"]
        n_l = "16 道异常样本包含未闭合 JSON 字符串\n（如 \"[{\\\"subject\\\": \\\"Math\\\"）或类型引号不一致\nSFT 拟合了训练集中的同类截断格式 (50%)\nRL 输出合法 JSON 因此无法匹配异常标签 (0%)"
        t_l = "(a) RLLA-4k 测试集拆分：64道正常样本 vs 16道异常样本"
        cats_r = ["Simple\n单工具 (400)", "Multiple\n多选一 (200)", "Parallel\n单工具并发 (200)", "Parallel-Multiple\n多选+并发 (200)"]
        n_r = "在多工具并发调用场景下差距拉大：\nParallel-Multiple 子集上\nPure RL (76.5%) 高于 Pure SFT (41.5%) 35.0 个百分点"
        t_r = "(b) BFCL-v3 域外标准集：四个子任务准确率对比"
        suptitle = "RLLA-4k 标注质量拆分与 BFCL-v3 各子集表现对比"
        suffix = "_zh"

    sft_l = [73.75, 79.69, 50.00]
    rl_l = [57.50, 71.88, 0.00]

    x1 = np.arange(len(cats_l))
    w1 = 0.30
    b_s1 = ax1.bar(x1 - w1 / 2, sft_l, w1, label="Pure SFT (2 Epochs)", color=C_SFT, zorder=3)
    b_r1 = ax1.bar(x1 + w1 / 2, rl_l, w1, label="Pure RL (8 Epochs)", color=C_RL, zorder=3)

    for bars, c in [(b_s1, C_SFT_DARK), (b_r1, C_RL_DARK)]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=9.8, fontweight="bold", color=c)

    ax1.annotate(
        n_l,
        xy=(2 + w1 / 2, 2.0),
        xytext=(0.75, 30.0),
        fontsize=8.8,
        color="#92400e",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fffbeb", edgecolor="#fcd34d"),
        arrowprops=dict(arrowstyle="->", color=C_WARN, lw=1.4),
    )

    ax1.set_title(t_l, fontsize=11.5, fontweight="bold", pad=12)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(cats_l, fontsize=10, fontweight="bold")
    ax1.set_ylabel("Strict Exact Match (%)", fontsize=10.5)
    ax1.set_ylim(0, 98)
    ax1.legend(loc="upper right", fontsize=9.5)

    sft_r = [69.25, 70.50, 51.50, 41.50]
    rl_r = [82.75, 81.50, 84.00, 76.50]

    x2 = np.arange(len(cats_r))
    b_s2 = ax2.bar(x2 - w1 / 2, sft_r, w1, label="Pure SFT (2 Epochs)", color=C_SFT, zorder=3)
    b_r2 = ax2.bar(x2 + w1 / 2, rl_r, w1, label="Pure RL (8 Epochs)", color=C_RL, zorder=3)

    for bars, c in [(b_s2, C_SFT_DARK), (b_r2, C_RL_DARK)]:
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=c)

    ax2.annotate(
        n_r,
        xy=(3 + w1 / 2, 76.5),
        xytext=(1.15, 88.0),
        fontsize=9.0,
        color=C_RL_DARK,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#ecfdf5", edgecolor="#6ee7b7"),
        arrowprops=dict(arrowstyle="->", color=C_RL, lw=1.4),
    )

    ax2.set_title(t_r, fontsize=11.5, fontweight="bold", pad=12)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(cats_r, fontsize=10, fontweight="bold")
    ax2.set_ylabel("Strict Exact Match (%)", fontsize=10.5)
    ax2.set_ylim(0, 105)
    ax2.legend(loc="upper left", fontsize=9.5)

    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = ASSETS_DIR / f"fig3_deep_dive_clean_dirty_and_bfcl{suffix}.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def fig4_reward_v2_staircase(lang: str = "en") -> Path:
    set_lang_font(lang)
    fig, ax = plt.subplots(figsize=(11.5, 5.0), facecolor="#ffffff")
    style_ax(ax)

    if lang == "en":
        stages = [
            "Stage 0\nMalformed JSON\nor Wrong Mode",
            "Stage 1\nValid XML Tags\n<think>+<tool_call>",
            "Stage 2\nTool Name Match\n+ Required Keys",
            "Stage 3\nCorrect Values\n(w/ Extra Keys)",
            "Stage 4\nNon-Exact Cap\n(Minor Type Diff)",
            "Stage 5\nStrict Exact Match\n(Exact Match = 1.0)",
        ]
        n1 = "Anti-Reward-Hacking 1 (Extra-Key Penalty):\nDeducts -0.75 per redundant parameter key\nnot present in Ground Truth"
        n2 = "Anti-Reward-Hacking 2 (Non-Exact Cap):\nCaps partial matches at +3.4 to preserve\na +0.60 margin for Strict Exact Match (+4.0)"
        title = "Dense Reward Shaping & Anti-Reward-Hacking Design in rlla_reward_v2.py"
        suffix = ""
    else:
        stages = [
            "阶段 0\nJSON解析失败\n或工具误调",
            "阶段 1\nXML标签合规\n<think>+<tool_call>",
            "阶段 2\n工具名正确\n+ 必填参数Key",
            "阶段 3\n参数值匹配\n(含冗余可选参数)",
            "阶段 4\n非全对分数上限\n(存在类型/局部偏差)",
            "阶段 5\n严格精确匹配\n(Exact Match = 1.0)",
        ]
        n1 = "抑制 Reward Hacking 1（冗余参数惩罚）：\n每多输出一个未出现在 GT 中的参数 Key\n扣减 0.75 分"
        n2 = "抑制 Reward Hacking 2（非全对封顶）：\n未完全匹配时最高封顶 +3.4 分\n与完全匹配 (+4.0 分) 保持 0.60 分差值"
        title = "rlla_reward_v2.py 的稠密过程奖励与抑制 Reward Hacking 设计"
        suffix = "_zh"

    scores = [-3.0, 1.0, 2.1, 2.38, 3.4, 4.0]
    colors = [C_DANGER, C_BASE, "#60a5fa", C_WARN, C_PURPLE, C_RL]

    x = np.arange(len(stages))
    bars = ax.bar(x, scores, width=0.50, color=colors, edgecolor="none", zorder=3)
    ax.axhline(0, color="#64748b", lw=1.2, zorder=2)

    for bar, s, c in zip(bars, scores, colors):
        va = "bottom" if s >= 0 else "top"
        offset = 0.15 if s >= 0 else -0.42
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            s + offset,
            f"{s:+.2f}",
            ha="center",
            va=va,
            fontsize=10.5,
            fontweight="bold",
            color=c if s < 0 else C_TEXT,
        )

    ax.annotate(
        n1,
        xy=(3, 2.38),
        xytext=(1.85, 3.55),
        fontsize=9.2,
        color="#92400e",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fffbeb", edgecolor="#fcd34d"),
        arrowprops=dict(arrowstyle="->", color=C_WARN, lw=1.4),
    )

    ax.annotate(
        n2,
        xy=(5, 4.0),
        xytext=(3.45, 4.30),
        fontsize=9.2,
        color=C_RL_DARK,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#ecfdf5", edgecolor="#6ee7b7"),
        arrowprops=dict(arrowstyle="->", color=C_RL, lw=1.4),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=9.8, fontweight="bold")
    ax.set_ylabel("Total Reward Score [-3.0, +4.0]", fontsize=10.5, fontweight="bold")
    ax.set_ylim(-3.8, 5.1)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)

    fig.tight_layout()
    out = ASSETS_DIR / f"fig4_reward_v2_staircase{suffix}.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


if __name__ == "__main__":
    for lang in ("en", "zh"):
        p1 = fig1_sft_vs_rl_overview(lang)
        p2 = fig2_training_dynamics_entropy_collapse(lang)
        p3 = fig3_deep_dive_clean_dirty_and_bfcl(lang)
        p4 = fig4_reward_v2_staircase(lang)
        print(f"Generated ({lang}):", p1.name, p2.name, p3.name, p4.name)
