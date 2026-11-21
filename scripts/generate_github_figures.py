#!/usr/bin/env python3
"""
Generate publication-grade (300 DPI) bilingual figures (English `fig*.png` & Chinese `fig*_zh.png`)
for GitHub README (`README.md` / `README_zh.md`) and technical reports (`docs/SFT_vs_RL_Tool_Use_Deep_Dive*.md`).
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

# Academic color palette
C_BASE = "#94a3b8"      # Slate 400
C_SFT1 = "#60a5fa"      # Blue 400
C_SFT = "#2563eb"       # Blue 600
C_SFT_DARK = "#1e40af"  # Blue 800
C_RL = "#059669"        # Emerald 600
C_RL_DARK = "#065f46"   # Emerald 800
C_COLD = "#0d9488"      # Teal 600
C_POS = "#7c3aed"       # Violet 600
C_POS_DARK = "#5b21b6"  # Violet 800
C_WARN = "#d97706"      # Amber 600
C_DANGER = "#dc2626"    # Red 600
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
    ax.tick_params(colors=C_TEXT, labelsize=9.8)


def fig1_sft_vs_rl_overview(lang: str = "en") -> Path:
    """Figure 1: Comprehensive 4-Benchmark Overview across 2,931 Tool-Calling Samples."""
    set_lang_font(lang)
    fig, ax = plt.subplots(figsize=(13.2, 6.0), facecolor="#ffffff")
    style_ax(ax)

    if lang == "en":
        benchmarks = [
            "RLLA-4k Test (N=80)\n[In-Domain · 20% Noisy GT]",
            "Expanded In-Domain (N=500)\n[100% Clean AST · Unseen Queries]",
            "BFCL-v3 Standard (N=1,000)\n[Out-of-Domain Benchmark]",
            "BFCL-v3 Live (N=1,351)\n[Real-World Multi-Candidate]",
        ]
        lbl_base = "0. Zero-Shot Base (Qwen3-4B-Base-rlla)"
        lbl_sft1 = "1a. Pure SFT (1.0 Ep · Min Eval-Loss=0.1582)"
        lbl_sft2 = "1b. Pure SFT (2.0 Ep · Min Train-Loss=0.0196)"
        lbl_pos = "2c. Cold-Start SFT (320) + GRPO (Step 100)"
        lbl_rl = "3b. Pure RL (GRPO 8.0 Ep · v2 Anti-Hacking)"
        note1 = "SFT Overfits Noisy GT in RLLA-80:\n73.8% on RLLA-80 (50% on broken GT),\nbut drops to 65.6% on Clean In-Domain 500\n(Pure RL reaches 63.2% without SFT)"
        note2 = "OOD Generalization (+21.1% Gain):\nPure RL (81.5%) & Cold-Start SFT+RL (81.2%)\noutperform Pure SFT (60.4%) by >20 points"
        note3 = "SFT Over-Training Penalty on BFCL-Live:\n1.0 Ep SFT (52.7%) > 2.0 Ep SFT (49.3%)\nCold-Start SFT + GRPO achieves highest 59.0%"
        ylabel = "Strict Exact Match (%)"
        title = "Comprehensive Evaluation Across 4 In-Domain & Out-of-Domain Benchmarks (N = 2,931 Tool-Calling Samples)"
        suffix = ""
    else:
        benchmarks = [
            "RLLA-4k 原测试集 (N=80)\n[同分布 · 含 20% 标注噪声]",
            "扩充同分布测试集 (N=500)\n[100% AST 校验 · 同分布未见题]",
            "BFCL-v3 标准集 (N=1,000)\n[分布外 OOD 权威基准]",
            "BFCL-v3 Live (N=1,351)\n[分布外真实用户多候选请求]",
        ]
        lbl_base = "0. Zero-Shot Base (未训练基座)"
        lbl_sft1 = "1a. Pure SFT (1.0 Ep · 验证集 Loss 最低点 0.1582)"
        lbl_sft2 = "1b. Pure SFT (2.0 Ep · 训练集 Loss 最低点 0.0196)"
        lbl_pos = "2c. Cold-Start SFT (320) + GRPO (Step 100 最优点)"
        lbl_rl = "3b. Pure RL (GRPO 8.0 Ep · v2 反作弊奖励)"
        note1 = "同分布扩充验证 (N=500)：\n剔除原 80 题的截断噪声后，SFT 为 65.6%\nPure RL (63.2%) 与 Cold-Start SFT+RL (62.6%)\n在零全量 SFT 下逼近同分布上限"
        note2 = "分布外泛化 (+21.1% 提升)：\nPure RL (81.5%) 与 Cold-Start SFT+RL (81.2%)\n大幅超越 Pure SFT (60.4%)"
        note3 = "SFT 过拟合代价 (BFCL-Live)：\n1.0 Ep SFT (52.7%) 显著优于 2.0 Ep SFT (49.3%)\nCold-Start SFT + GRPO 取得全场最高 59.0%"
        ylabel = "Strict Exact Match (%)"
        title = "四大域内与域外基准（共 2,931 道工具调用题）五大代表模型严格精确匹配率对比"
        suffix = "_zh"

    base_s = [43.75, 52.20, 69.50, 57.81]
    sft1_s = [73.75, 64.60, 60.20, 52.70]
    sft2_s = [73.75, 65.60, 60.40, 49.30]
    pos_s = [50.00, 62.60, 81.20, 58.99]
    rl_s = [56.25, 63.20, 81.50, 58.77]

    x = np.arange(len(benchmarks))
    w = 0.155

    b0 = ax.bar(x - 2 * w, base_s, w, label=lbl_base, color=C_BASE, zorder=3, alpha=0.88)
    b1 = ax.bar(x - w, sft1_s, w, label=lbl_sft1, color=C_SFT1, zorder=3)
    b2 = ax.bar(x, sft2_s, w, label=lbl_sft2, color=C_SFT, zorder=3)
    b3 = ax.bar(x + w, pos_s, w, label=lbl_pos, color=C_POS, zorder=3)
    b4 = ax.bar(x + 2 * w, rl_s, w, label=lbl_rl, color=C_RL, zorder=3)

    for bars, color in [(b0, C_SUB), (b1, C_SFT_DARK), (b2, C_SFT_DARK), (b3, C_POS_DARK), (b4, C_RL_DARK)]:
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 1.0,
                f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=8.3,
                fontweight="bold",
                color=color,
            )

    ax.annotate(
        note1,
        xy=(1, 65.6),
        xytext=(0.12, 83.5),
        fontsize=8.7,
        color=C_SFT_DARK,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#eff6ff", edgecolor="#93c5fd", lw=1.1),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.12", color=C_SFT, lw=1.3),
    )

    ax.annotate(
        note2,
        xy=(2 + 2 * w, 81.5),
        xytext=(1.62, 90.0),
        fontsize=8.7,
        color=C_RL_DARK,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#ecfdf5", edgecolor="#6ee7b7", lw=1.1),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.12", color=C_RL, lw=1.3),
    )

    ax.annotate(
        note3,
        xy=(3 + w, 59.0),
        xytext=(2.38, 71.5),
        fontsize=8.7,
        color=C_POS_DARK,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f3ff", edgecolor="#c4b5fd", lw=1.1),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.12", color=C_POS, lw=1.3),
    )

    ax.set_ylabel(ylabel, fontsize=11, color=C_TEXT, fontweight="bold")
    ax.set_title(title, fontsize=12.5, fontweight="bold", color=C_TEXT, pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 106)
    ax.legend(loc="upper left", ncol=2, frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1", fontsize=8.8)

    fig.tight_layout()
    out = ASSETS_DIR / f"fig1_sft_vs_rl_overview{suffix}.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def fig2_training_dynamics_entropy_collapse(lang: str = "en") -> Path:
    """Figure 2: Step 1 (SFT Loss & Checkpoint Selection) + Step 2 & 3 (RL Val Reward & Policy Entropy)."""
    set_lang_font(lang)
    sft_log_path = PROJECT_ROOT / "outputs" / "sft" / "sft-rlla4k-Qwen3-4B-Base-rlla-ep2" / "trainer_log.jsonl"
    metrics_dir = PROJECT_ROOT / "logs" / "metrics"
    purerl_log = metrics_dir / "grpo-rlla4k-Qwen3-4B-Base-rlla-n4g8_20261019_190110.jsonl"
    sftrl_log = metrics_dir / "grpo-rlla4k-Qwen3-4B-SFT-ep2-n4g8_20261020_104244.jsonl"
    coldrl_log = metrics_dir / "grpo-rlla4k-v2reward-ColdStartSFT320-n4g8_20261121_104610.jsonl"

    sft_tr_s, sft_tr_l, sft_ev_s, sft_ev_l = [], [], [], []
    for line in open(sft_log_path):
        r = json.loads(line)
        if "loss" in r:
            sft_tr_s.append(r["current_steps"])
            sft_tr_l.append(r["loss"])
        if "eval_loss" in r:
            sft_ev_s.append(r["current_steps"])
            sft_ev_l.append(r["eval_loss"])

    def parse_rl_log(path: Path):
        vs, vr, ts, ent = [], [], [], []
        for line in open(path):
            row = json.loads(line)
            step = row.get("step", 0)
            data = row.get("data", {})
            if "val-core/rlla/reward/mean@1" in data:
                vs.append(step)
                vr.append(data["val-core/rlla/reward/mean@1"])
            if "actor/entropy" in data:
                ts.append(step)
                ent.append(data["actor/entropy"])
        return vs, vr, ts, ent

    p_vs, p_vr, p_ts, p_ent = parse_rl_log(purerl_log)
    s_vs, s_vr, s_ts, s_ent = parse_rl_log(sftrl_log)
    c_vs, c_vr, c_ts, c_ent = parse_rl_log(coldrl_log)

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(16.5, 5.2), facecolor="#ffffff")
    for ax in (ax0, ax1, ax2):
        style_ax(ax)

    if lang == "en":
        t0 = "(a) Step 1: SFT Train vs. Eval Loss (Checkpoint Selection)"
        l_tr = "SFT Train Loss (0.785 -> 0.020)"
        l_ev = "SFT Eval Loss (Min 0.1582 @ 1.0 Ep)"
        n0_1 = "1.0 Ep (Step 62):\nGlobal Min eval_loss = 0.1582\nBest OOD (BFCL-Live: 52.7%)"
        n0_2 = "2.0 Ep (Step 124):\neval_loss rebounds to 0.1700\nOOD drops to 49.3% (-3.4%)"
        t1 = "(b) Step 2 & 3: GRPO Validation Reward Progression"
        l_sft = "Full-SFT + GRPO (+2.96 -> +2.94, Stalled)"
        l_cold = "Cold-Start SFT + GRPO (+2.51 -> +2.75 @ Step 100)"
        l_rl = "Pure RL (+0.96 -> +2.61 @ Step 120)"
        n1_c = "Cold-Start SFT + GRPO Peak:\nStep 100 reaches +2.7486\n(+0.24 over Step 0)"
        n1_s = "Full-SFT + GRPO Stalls:\n75/80 outputs identical to Step 0"
        t2 = "(c) Step 3 Mechanism: Token-Level Policy Entropy"
        l_ent_rl = "Pure RL Entropy (0.72 -> 0.25)"
        l_ent_cold = "Cold-Start SFT + GRPO Entropy (0.49 -> 0.21)"
        l_ent_sft = "Full-SFT + GRPO Entropy (~0.08, Collapsed)"
        n2 = "Entropy Collapse vs. Preservation:\nFull-SFT collapses entropy to 0.08 (std(r)=0);\nCold-Start SFT (320) preserves entropy at 0.49,\nenabling effective GRPO exploration!"
        suptitle = "Three-Stage Experimental Progression: SFT Overfitting Point, RL Convergence Peak, and SFT+RL Entropy Dynamics"
        suffix = ""
    else:
        t0 = "(a) 第一步：SFT 训练与验证 Loss 曲线（最优 Checkpoint 选择）"
        l_tr = "SFT 训练 Loss (0.785 -> 0.020)"
        l_ev = "SFT 验证集 Loss (最低点 0.1582 @ 1.0 Ep)"
        n0_1 = "1.0 Ep (Step 62)：\n验证集 Loss 最低 (0.1582)\n域外泛化最优 (BFCL-Live: 52.7%)"
        n0_2 = "2.0 Ep (Step 124)：\n验证集 Loss 反弹至 0.1700\n域外跌至 49.3% (-3.4%)"
        t1 = "(b) 第二/三步：GRPO 验证集平均奖励演进曲线"
        l_sft = "Full-SFT + GRPO (+2.96 -> +2.94, 停滞)"
        l_cold = "Cold-Start SFT + GRPO (+2.51 -> +2.75 @ Step 100)"
        l_rl = "Pure RL (+0.96 -> +2.61 @ Step 120)"
        n1_c = "Cold-Start SFT + GRPO 最优点：\nStep 100 达到峰值 +2.7486\n（较 Step 0 提升 +0.24 分）"
        n1_s = "Full-SFT + GRPO 完全停滞：\n验证集 75/80 题输出与 Step 0 完全相同"
        t2 = "(c) 第三步机理：Token 级策略熵演进 (actor/entropy)"
        l_ent_rl = "Pure RL 策略熵 (0.72 -> 0.25)"
        l_ent_cold = "Cold-Start SFT + GRPO 策略熵 (0.49 -> 0.21)"
        l_ent_sft = "Full-SFT + GRPO 策略熵 (~0.08, 坍缩)"
        n2 = "策略熵坍缩 vs 保留对比：\n全量伪 CoT SFT 将策略熵压死至 0.08 (std(r)=0)\n微缩真 CoT 冷启动 (320条) 保留策略熵 0.49，\n使后续 GRPO 获得显著正增益！"
        suptitle = "三阶段实验演进：SFT 过拟合拐点、RL 收敛峰值与 SFT+RL 策略熵机理"
        suffix = "_zh"

    # Panel (a): SFT Loss
    ax0.plot(sft_tr_s, sft_tr_l, lw=2.0, color=C_SFT1, alpha=0.85, label=l_tr)
    ax0.plot(sft_ev_s, sft_ev_l, marker="o", markersize=6, lw=2.4, color=C_SFT_DARK, label=l_ev)
    ax0.axvline(62, color=C_RL, linestyle="--", lw=1.4, alpha=0.8)
    ax0.annotate(
        n0_1,
        xy=(62, 0.1582),
        xytext=(15, 0.42),
        fontsize=8.6,
        color=C_RL_DARK,
        bbox=dict(boxstyle="round,pad=0.38", facecolor="#ecfdf5", edgecolor="#6ee7b7"),
        arrowprops=dict(arrowstyle="->", color=C_RL, lw=1.3),
    )
    ax0.annotate(
        n0_2,
        xy=(124, 0.1700),
        xytext=(58, 0.62),
        fontsize=8.6,
        color="#991b1b",
        bbox=dict(boxstyle="round,pad=0.38", facecolor="#fef2f2", edgecolor="#fca5a5"),
        arrowprops=dict(arrowstyle="->", color=C_DANGER, lw=1.3),
    )
    ax0.set_title(t0, fontsize=10.8, fontweight="bold", pad=10)
    ax0.set_xlabel("SFT Steps (62 Steps = 1.0 Epoch, 124 Steps = 2.0 Epochs)", fontsize=9.8)
    ax0.set_ylabel("Cross-Entropy Loss", fontsize=9.8)
    ax0.set_ylim(0.0, 0.86)
    ax0.legend(loc="upper right", fontsize=8.6, frameon=True, facecolor="#f8fafc")

    # Panel (b): RL Validation Reward
    ax1.plot(s_vs, s_vr, marker="s", markersize=4, lw=2.0, color=C_DANGER, label=l_sft)
    ax1.plot(c_vs, c_vr, marker="^", markersize=4.8, lw=2.3, color=C_POS, label=l_cold)
    ax1.plot(p_vs, p_vr, marker="o", markersize=4, lw=2.0, color=C_RL, label=l_rl)
    ax1.axvline(100, color=C_POS, linestyle="--", lw=1.3, alpha=0.7)
    ax1.annotate(
        n1_c,
        xy=(100, 2.7486),
        xytext=(28, 2.12),
        fontsize=8.6,
        color=C_POS_DARK,
        bbox=dict(boxstyle="round,pad=0.38", facecolor="#f5f3ff", edgecolor="#c4b5fd"),
        arrowprops=dict(arrowstyle="->", color=C_POS, lw=1.3),
    )
    ax1.annotate(
        n1_s,
        xy=(60, 2.94),
        xytext=(8, 3.12),
        fontsize=8.4,
        color="#991b1b",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#fef2f2", edgecolor="#fca5a5"),
        arrowprops=dict(arrowstyle="->", color=C_DANGER, lw=1.2),
    )
    ax1.set_title(t1, fontsize=10.8, fontweight="bold", pad=10)
    ax1.set_xlabel("GRPO Training Steps (30 Steps = 1.0 Epoch)", fontsize=9.8)
    ax1.set_ylabel("Validation Mean Reward [-3.0, +4.0]", fontsize=9.8)
    ax1.set_ylim(0.8, 3.42)
    ax1.legend(loc="lower right", fontsize=8.4, frameon=True, facecolor="#f8fafc")

    # Panel (c): Policy Entropy
    ax2.plot(p_ts, p_ent, lw=1.9, color=C_RL, alpha=0.88, label=l_ent_rl)
    ax2.plot(c_ts, c_ent, lw=2.1, color=C_POS, alpha=0.92, label=l_ent_cold)
    ax2.plot(s_ts, s_ent, lw=2.2, color=C_DANGER, alpha=0.92, label=l_ent_sft)
    ax2.annotate(
        n2,
        xy=(25, 0.36),
        xytext=(18, 0.50),
        fontsize=8.5,
        color=C_POS_DARK,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f3ff", edgecolor="#c4b5fd"),
        arrowprops=dict(arrowstyle="->", color=C_POS, lw=1.3),
    )
    ax2.set_title(t2, fontsize=10.8, fontweight="bold", pad=10)
    ax2.set_xlabel("GRPO Training Steps (Global Step)", fontsize=9.8)
    ax2.set_ylabel("Token-Level Policy Entropy", fontsize=9.8)
    ax2.set_ylim(0.0, 0.82)
    ax2.legend(loc="upper right", fontsize=8.4, frameon=True, facecolor="#f8fafc")

    fig.suptitle(suptitle, fontsize=12.8, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = ASSETS_DIR / f"fig2_training_dynamics_entropy_collapse{suffix}.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def fig3_deep_dive_clean_dirty_and_bfcl(lang: str = "en") -> Path:
    """Figure 3: In-Domain Clean vs Noisy Split (N=500 & N=80) and BFCL-v3 Subsets."""
    set_lang_font(lang)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.5), facecolor="#ffffff")
    style_ax(ax1)
    style_ax(ax2)

    if lang == "en":
        cats_l = [
            "RLLA-80 Clean\n(N=64 · Valid GT)",
            "RLLA-80 Noisy\n(N=16 · Broken GT)",
            "InDom-500 xLAM\n(N=250 · Clean Types)",
            "InDom-500 ToolACE\n(N=250 · Quoted Ints)",
        ]
        n_l = "Why SFT Leads on Raw In-Domain:\nSFT memorizes broken/quoted-int GT conventions\n(50.0% on RLLA-16; 47.6% on ToolACE-250).\nOn Clean xLAM-250, Pure RL reaches 82.8%\nand Cold-Start SFT+RL reaches 82.4%!"
        t_l = "(a) In-Domain Label & Schema Audit (RLLA-80 & InDom-500)"
        cats_r = ["Simple\n(N=400)", "Multiple\n(N=200)", "Parallel\n(N=200)", "Parallel-Multiple\n(N=200)"]
        n_r = "Multi-Tool Composition Scaling:\nOn Parallel-Multiple, Pure RL (76.5%) &\nCold-Start SFT+RL (75.5%) beat SFT (41.5%) by +35%"
        t_r = "(b) BFCL-v3 Out-of-Domain Breakdown by Complexity"
        suptitle = "Deconstructing In-Domain Label Artifacts vs. Out-of-Domain Compositional Generalization"
        suffix = ""
    else:
        cats_l = [
            "RLLA-80 正常题\n(N=64 · 合法GT)",
            "RLLA-80 异常题\n(N=16 · 截断GT)",
            "扩充集 xLAM\n(N=250 · 标准类型)",
            "扩充集 ToolACE\n(N=250 · 数值加引号)",
        ]
        n_l = "拆解 SFT 的同分布表面优势：\nSFT 记住了截断 JSON (50.0%) 与引号数值习惯。\n在类型规范的同分布 xLAM (250题) 上，\nPure RL (82.8%) 与 Cold-Start SFT+RL (82.4%)\n与 SFT (83.6%) 几乎完全持平！"
        t_l = "(a) 同分布测试集质量审计（RLLA-80 与扩充集 500 题）"
        cats_r = ["Simple\n单工具 (400)", "Multiple\n多选一 (200)", "Parallel\n单工具并发 (200)", "Parallel-Multiple\n多选+并发 (200)"]
        n_r = "复杂度越高 RL 优势越大：\n在 Parallel-Multiple 子集上，\nPure RL (76.5%) 与 Cold-Start SFT+RL (75.5%)\n领先 Pure SFT (41.5%) 达 +35 个百分点"
        t_r = "(b) BFCL-v3 域外标准集四个难度子任务对比"
        suptitle = "同分布标注规范性拆解（RLLA-80 / 扩充集 500 题）与域外组合复杂度扩展（BFCL-v3）"
        suffix = "_zh"

    sft_l = [79.69, 50.00, 83.60, 47.60]
    pos_l = [62.50, 0.00, 82.40, 42.80]
    rl_l = [71.88, 0.00, 82.80, 43.60]

    x1 = np.arange(len(cats_l))
    w = 0.24
    b_s1 = ax1.bar(x1 - w, sft_l, w, label="1b. Pure SFT (2.0 Ep)", color=C_SFT, zorder=3)
    b_p1 = ax1.bar(x1, pos_l, w, label="2c. Cold-Start SFT + GRPO", color=C_POS, zorder=3)
    b_r1 = ax1.bar(x1 + w, rl_l, w, label="3b. Pure RL (8.0 Ep, v2)", color=C_RL, zorder=3)

    for bars, c in [(b_s1, C_SFT_DARK), (b_p1, C_POS_DARK), (b_r1, C_RL_DARK)]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.2, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.4, fontweight="bold", color=c)

    ax1.annotate(
        n_l,
        xy=(2 + w, 82.8),
        xytext=(0.55, 86.0),
        fontsize=8.5,
        color=C_RL_DARK,
        bbox=dict(boxstyle="round,pad=0.38", facecolor="#ecfdf5", edgecolor="#6ee7b7"),
        arrowprops=dict(arrowstyle="->", color=C_RL, lw=1.3),
    )

    ax1.set_title(t_l, fontsize=11.2, fontweight="bold", pad=12)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(cats_l, fontsize=9.5, fontweight="bold")
    ax1.set_ylabel("Strict Exact Match (%)", fontsize=10.5)
    ax1.set_ylim(0, 108)
    ax1.legend(loc="upper left", fontsize=8.6)

    sft_r = [67.50, 61.00, 64.50, 41.50]
    pos_r = [82.00, 82.00, 84.50, 75.50]
    rl_r = [82.75, 81.50, 84.00, 76.50]

    x2 = np.arange(len(cats_r))
    b_s2 = ax2.bar(x2 - w, sft_r, w, label="1b. Pure SFT (2.0 Ep)", color=C_SFT, zorder=3)
    b_p2 = ax2.bar(x2, pos_r, w, label="2c. Cold-Start SFT + GRPO", color=C_POS, zorder=3)
    b_r2 = ax2.bar(x2 + w, rl_r, w, label="3b. Pure RL (8.0 Ep, v2)", color=C_RL, zorder=3)

    for bars, c in [(b_s2, C_SFT_DARK), (b_p2, C_POS_DARK), (b_r2, C_RL_DARK)]:
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 1.2, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.4, fontweight="bold", color=c)

    ax2.annotate(
        n_r,
        xy=(3 + w, 76.5),
        xytext=(1.05, 90.0),
        fontsize=8.6,
        color=C_RL_DARK,
        bbox=dict(boxstyle="round,pad=0.38", facecolor="#ecfdf5", edgecolor="#6ee7b7"),
        arrowprops=dict(arrowstyle="->", color=C_RL, lw=1.3),
    )

    ax2.set_title(t_r, fontsize=11.2, fontweight="bold", pad=12)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(cats_r, fontsize=9.5, fontweight="bold")
    ax2.set_ylabel("Strict Exact Match (%)", fontsize=10.5)
    ax2.set_ylim(0, 108)
    ax2.legend(loc="upper left", fontsize=8.6)

    fig.suptitle(suptitle, fontsize=12.8, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = ASSETS_DIR / f"fig3_deep_dive_clean_dirty_and_bfcl{suffix}.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def fig4_reward_v2_staircase(lang: str = "en") -> Path:
    """Figure 4: Dense Reward Shaping & Error Taxonomy Reduction (v1 -> v2 & SFT vs RL)."""
    set_lang_font(lang)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.4), facecolor="#ffffff")
    style_ax(ax1)
    style_ax(ax2)

    if lang == "en":
        stages = [
            "Stage 0\nMalformed JSON\nor Wrong Mode",
            "Stage 1\nValid XML Tags\n<think>+<tool_call>",
            "Stage 2\nTool Name\n+ Req Keys",
            "Stage 3\nCorrect Vals\n(w/ Extra Key)",
            "Stage 4\nNon-Exact Cap\n(Minor Diff)",
            "Stage 5\nStrict Exact\n(EM = 1.0)",
        ]
        n1 = "Anti-Hacking 1 (-0.75/extra key):\nCuts E5 Extra-Key Hallucination on\nInDom-500 by -28% (50 -> 36)"
        n2 = "Anti-Hacking 2 (+3.4 Cap):\nPreserves +0.60 margin for Exact Match"
        t1 = "(a) 6-Stage Reward Staircase in rlla_reward_v2.py"
        err_cats = [
            "E7: Type/Quote\nMismatch\n(BFCL-1k)",
            "E4: Missing\nReq Param\n(BFCL-1k)",
            "E2: False\nAbstention\n(BFCL-Live)",
            "E5: Extra Key\nHallucination\n(InDom-500)",
        ]
        t2 = "(b) Error Count Comparison Across Failure Modes (Lower is Better)"
        n_err = "Key Error Taxonomy Findings:\n- SFT suffers 7.6x more Type/Quote errors (E7: 146 vs 19)\n- 2.0 Ep SFT triggers 49 False Abstentions (E2) on BFCL-Live\n- v2 Reward cuts RL Extra-Key errors (E5) from 50 -> 36"
        suptitle = "Reward Engineering (rlla_reward_v2.py) & Quantitative Error Taxonomy Across Models"
        suffix = ""
    else:
        stages = [
            "阶段 0\nJSON解析失败\n或模式误判",
            "阶段 1\nXML标签合规\n<think>+<tool_call>",
            "阶段 2\n工具名正确\n+ 必填Key",
            "阶段 3\n参数值匹配\n(含冗余Key)",
            "阶段 4\n非全对封顶\n(局部偏差)",
            "阶段 5\n严格全对\n(EM = 1.0)",
        ]
        n1 = "反作弊机制 1 (-0.75/冗余Key)：\n将扩充集 500 题上的冗余参数错误 (E5)\n从 50 题压降至 36 题 (-28%)"
        n2 = "反作弊机制 2 (+3.4 封顶)：\n与全对 (+4.0) 保持 +0.60 净激励差"
        t1 = "(a) rlla_reward_v2.py 六阶奖励阶梯与反作弊设计"
        err_cats = [
            "E7: 类型/引号\n污染错误\n(BFCL-1k)",
            "E4: 漏填必填\n参数错误\n(BFCL-1k)",
            "E2: 过度拒答\n模式错误\n(BFCL-Live)",
            "E5: 幻觉多填\n可选参数\n(InDom-500)",
        ]
        t2 = "(b) 四类核心错误频次对比（越低越好）"
        n_err = "错误分类学核心发现：\n- SFT 的类型/引号污染错误 (E7) 是 RL 的 7.6 倍 (146 vs 19)\n- 2.0 Ep SFT 在 Live 上产生 49 例过度拒答 (E2, 1.0 Ep 仅 5 例)\n- v2 奖励使 RL 的多填可选参数错误 (E5) 下降 28% (50 -> 36)"
        suptitle = "奖励函数反作弊设计 (rlla_reward_v2.py) 与跨模型错误分类学 (Error Taxonomy) 定量对比"
        suffix = "_zh"

    scores = [-3.0, 1.0, 2.1, 2.38, 3.4, 4.0]
    colors = [C_DANGER, C_BASE, C_SFT1, C_WARN, C_POS, C_RL]

    x1 = np.arange(len(stages))
    bars = ax1.bar(x1, scores, width=0.50, color=colors, edgecolor="none", zorder=3)
    ax1.axhline(0, color="#64748b", lw=1.2, zorder=2)

    for bar, s, c in zip(bars, scores, colors):
        va = "bottom" if s >= 0 else "top"
        offset = 0.14 if s >= 0 else -0.42
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            s + offset,
            f"{s:+.2f}",
            ha="center",
            va=va,
            fontsize=9.6,
            fontweight="bold",
            color=c if s < 0 else C_TEXT,
        )

    ax1.annotate(
        n1,
        xy=(3, 2.38),
        xytext=(1.2, 3.55),
        fontsize=8.5,
        color="#92400e",
        bbox=dict(boxstyle="round,pad=0.38", facecolor="#fffbeb", edgecolor="#fcd34d"),
        arrowprops=dict(arrowstyle="->", color=C_WARN, lw=1.3),
    )
    ax1.annotate(
        n2,
        xy=(5, 4.0),
        xytext=(3.1, 4.42),
        fontsize=8.5,
        color=C_RL_DARK,
        bbox=dict(boxstyle="round,pad=0.38", facecolor="#ecfdf5", edgecolor="#6ee7b7"),
        arrowprops=dict(arrowstyle="->", color=C_RL, lw=1.3),
    )

    ax1.set_xticks(x1)
    ax1.set_xticklabels(stages, fontsize=8.8, fontweight="bold")
    ax1.set_ylabel("Total Reward Score [-3.0, +4.0]", fontsize=10.2, fontweight="bold")
    ax1.set_ylim(-3.8, 5.2)
    ax1.set_title(t1, fontsize=11.0, fontweight="bold", pad=12)

    # Panel (b): Error Taxonomy comparison
    e_sft1 = [146, 69, 5, 11]
    e_sft2 = [137, 52, 49, 14]
    e_rlv1 = [24, 6, 2, 50]
    e_pos = [18, 6, 0, 39]
    e_rlv2 = [19, 8, 3, 36]

    x2 = np.arange(len(err_cats))
    w = 0.15
    b_e1 = ax2.bar(x2 - 2 * w, e_sft1, w, label="1a. Pure SFT (1.0 Ep)", color=C_SFT1, zorder=3)
    b_e2 = ax2.bar(x2 - w, e_sft2, w, label="1b. Pure SFT (2.0 Ep)", color=C_SFT, zorder=3)
    b_e3 = ax2.bar(x2, e_rlv1, w, label="3a. Pure RL (4.0 Ep, v1)", color=C_WARN, zorder=3)
    b_e4 = ax2.bar(x2 + w, e_pos, w, label="2c. Cold-Start SFT + GRPO (v2)", color=C_POS, zorder=3)
    b_e5 = ax2.bar(x2 + 2 * w, e_rlv2, w, label="3b. Pure RL (8.0 Ep, v2)", color=C_RL, zorder=3)

    for bars, c in [(b_e1, C_SFT_DARK), (b_e2, C_SFT_DARK), (b_e3, "#92400e"), (b_e4, C_POS_DARK), (b_e5, C_RL_DARK)]:
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 2.0, f"{int(h)}", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color=c)

    ax2.annotate(
        n_err,
        xy=(0 - w, 137),
        xytext=(0.82, 118),
        fontsize=8.4,
        color=C_TEXT,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8fafc", edgecolor="#94a3b8"),
    )

    ax2.set_xticks(x2)
    ax2.set_xticklabels(err_cats, fontsize=9.2, fontweight="bold")
    ax2.set_ylabel("Number of Error Cases (Count)", fontsize=10.2, fontweight="bold")
    ax2.set_ylim(0, 175)
    ax2.set_title(t2, fontsize=11.0, fontweight="bold", pad=12)
    ax2.legend(loc="upper right", fontsize=8.2, frameon=True, facecolor="#f8fafc")

    fig.suptitle(suptitle, fontsize=12.8, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
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
