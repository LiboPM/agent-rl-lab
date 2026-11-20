#!/usr/bin/env python3
"""
Generate publication-grade (300 DPI) English figures for GitHub README and technical report.
Saves PNG files to `assets/`.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
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


def style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor("#ffffff")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.grid(axis="y", linestyle="--", alpha=0.45, color="#cbd5e1", zorder=0)
    ax.tick_params(colors=C_TEXT, labelsize=10)


def fig1_sft_vs_rl_overview() -> Path:
    """Figure 1: 3-Benchmark Comparison (In-Domain RLLA vs OOD BFCL-v3 vs OOD BFCL-Live)."""
    fig, ax = plt.subplots(figsize=(11.5, 5.8), facecolor="#ffffff")
    style_ax(ax)

    benchmarks = [
        "RLLA-4k Test (N=80)\n[In-Domain Split]",
        "BFCL-v3 Standard (N=1,000)\n[Out-of-Domain Benchmark]",
        "BFCL-v3 Live (N=1,351)\n[Real-World User Queries]",
    ]
    base_scores = [43.75, 60.20, 45.30]
    sft_scores = [73.75, 60.40, 49.30]
    rl_scores = [57.50, 81.50, 58.77]

    x = np.arange(len(benchmarks))
    width = 0.24

    b0 = ax.bar(x - width, base_scores, width, label="Zero-Shot Base (Qwen3-4B-Base-rlla)", color=C_BASE, edgecolor="none", zorder=3, alpha=0.85)
    b1 = ax.bar(x, sft_scores, width, label="Pure SFT (2 Epochs · 8 GPUs, 10m)", color=C_SFT, edgecolor="none", zorder=3)
    b2 = ax.bar(x + width, rl_scores, width, label="Pure RL (GRPO 8 Epochs · v2 Reward)", color=C_RL, edgecolor="none", zorder=3)

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
        "In-Domain Fit: Pure SFT leads (+16.3%)\nin 10 mins (18x faster than RL)\n*On 64 clean samples: RL reaches 71.9%",
        xy=(0, 73.75),
        xytext=(-0.38, 86.5),
        fontsize=9.2,
        color=C_SFT_DARK,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#eff6ff", edgecolor="#93c5fd", lw=1.2),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.15", color=C_SFT, lw=1.4),
    )

    ax.annotate(
        "OOD Generalization: Pure RL leads by +21.1%\nSFT stagnates near Base (60.2% -> 60.4%)\nRL generalizes to unseen schemas (81.5%)",
        xy=(1 + width, 81.50),
        xytext=(0.70, 89.5),
        fontsize=9.2,
        color=C_RL_DARK,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#ecfdf5", edgecolor="#6ee7b7", lw=1.2),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.15", color=C_RL, lw=1.4),
    )

    ax.annotate(
        "Real-World Robustness: Pure RL +9.5%\n(10-20 candidate tools per prompt)",
        xy=(2 + width, 58.77),
        xytext=(1.60, 72.5),
        fontsize=9.2,
        color=C_RL_DARK,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#ecfdf5", edgecolor="#6ee7b7", lw=1.2),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.15", color=C_RL, lw=1.4),
    )

    ax.set_ylabel("Strict Exact Match (%)", fontsize=11, color=C_TEXT, fontweight="bold")
    ax.set_title(
        "In-Domain Memorization (SFT) vs. Out-of-Domain Generalization (RL) Across 2,431 Samples",
        fontsize=13,
        fontweight="bold",
        color=C_TEXT,
        pad=16,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, fontsize=10.5, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1", fontsize=9.8)

    fig.tight_layout()
    out = ASSETS_DIR / "fig1_sft_vs_rl_overview.png"
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4), facecolor="#ffffff")
    style_ax(ax1)
    style_ax(ax2)

    # Panel A: Validation Reward Curve
    ax1.plot(s_vs, s_vr, marker="s", markersize=4.5, lw=2.2, color=C_DANGER, label="Full-SFT + GRPO (+2.96 -> +2.94, Stalled)")
    ax1.plot(p_vs, p_vr, marker="o", markersize=4.5, lw=2.2, color=C_RL, label="Pure RL (+0.96 -> +2.61, 4 Epochs)")
    ax1.axhline(2.676, color=C_RL_DARK, linestyle=":", lw=1.8, label="Pure RL + v2 Reward (Epoch 8: +2.68)")

    ax1.annotate(
        "Reward Stagnation in Full-SFT + GRPO:\n75/80 validation outputs at Step 120 are\ncharacter-for-character identical to Step 0",
        xy=(80, 2.94),
        xytext=(30, 2.12),
        fontsize=9.2,
        color="#991b1b",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fef2f2", edgecolor="#fca5a5"),
        arrowprops=dict(arrowstyle="->", color=C_DANGER, lw=1.4),
    )
    ax1.annotate(
        "Pure RL climbs steadily (+1.72)\n97.5% samples preserve genuine CoT",
        xy=(120, 2.61),
        xytext=(52, 1.32),
        fontsize=9.2,
        color=C_RL_DARK,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#ecfdf5", edgecolor="#6ee7b7"),
        arrowprops=dict(arrowstyle="->", color=C_RL, lw=1.4),
    )

    ax1.set_title("(a) Validation Reward Progression (val-core/rlla/reward/mean@1)", fontsize=11.5, fontweight="bold", pad=12)
    ax1.set_xlabel("GRPO Training Steps (Global Step)", fontsize=10.5)
    ax1.set_ylabel("Validation Mean Reward [-3.0, +4.0]", fontsize=10.5)
    ax1.set_ylim(0.7, 3.35)
    ax1.legend(loc="lower right", fontsize=9.2, frameon=True, facecolor="#f8fafc")

    # Panel B: Policy Entropy Curve
    ax2.plot(p_ts, p_ent, lw=2.0, color=C_RL, alpha=0.9, label="Pure RL Policy Entropy (0.72 -> 0.25)")
    ax2.plot(s_ts, s_ent, lw=2.2, color=C_DANGER, alpha=0.9, label="Full-SFT + GRPO Policy Entropy (~0.08)")

    ax2.annotate(
        "Root Cause — Policy Entropy Collapse (0.08):\nBoilerplate <think> in SFT data overwrites CoT;\nall G=8 rollouts become identical -> std(r)=0 -> zero gradient",
        xy=(30, 0.085),
        xytext=(16, 0.36),
        fontsize=9.2,
        color="#991b1b",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#fef2f2", edgecolor="#fca5a5"),
        arrowprops=dict(arrowstyle="->", color=C_DANGER, lw=1.4),
    )

    ax2.set_title("(b) Token-Level Policy Entropy (actor/entropy)", fontsize=11.5, fontweight="bold", pad=12)
    ax2.set_xlabel("GRPO Training Steps (Global Step)", fontsize=10.5)
    ax2.set_ylabel("Policy Entropy", fontsize=10.5)
    ax2.set_ylim(0.0, 0.82)
    ax2.legend(loc="upper right", fontsize=9.2, frameon=True, facecolor="#f8fafc")

    fig.suptitle(
        "Training Dynamics: Why Full-SFT + GRPO Stalls Due to Boilerplate CoT & Entropy Collapse",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = ASSETS_DIR / "fig2_training_dynamics_entropy_collapse.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def fig3_deep_dive_clean_dirty_and_bfcl() -> Path:
    """Figure 3: RLLA Data Quality Breakdown (64 Clean vs 16 Dirty) & BFCL 4 Subcategories."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4), facecolor="#ffffff")
    style_ax(ax1)
    style_ax(ax2)

    # Left: RLLA-4k Clean vs Dirty Breakdown
    cats_l = [
        "All Test Samples\n(N=80)",
        "Clean Samples\n(N=64 · 80%)",
        "Corrupted GT\n(N=16 · 20%)",
    ]
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
        "16 Corrupted GTs have truncated unclosed JSON\n(e.g. \"[{\\\"subject\\\": \\\"Math\\\") or quoted ints.\nSFT memorizes the dataset truncation bug (50%);\nRL emits valid JSON and scores 0% on broken GT.",
        xy=(2 + w1 / 2, 2.0),
        xytext=(0.75, 30.0),
        fontsize=8.8,
        color="#92400e",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fffbeb", edgecolor="#fcd34d"),
        arrowprops=dict(arrowstyle="->", color=C_WARN, lw=1.4),
    )

    ax1.set_title("(a) RLLA-4k Audit: 64 Clean vs. 16 Corrupted GTs", fontsize=11.5, fontweight="bold", pad=12)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(cats_l, fontsize=10, fontweight="bold")
    ax1.set_ylabel("Strict Exact Match (%)", fontsize=10.5)
    ax1.set_ylim(0, 98)
    ax1.legend(loc="upper right", fontsize=9.5)

    # Right: BFCL-v3 4 Subcategories
    cats_r = [
        "Simple\n(N=400)",
        "Multiple\n(N=200)",
        "Parallel\n(N=200)",
        "Parallel-Multiple\n(N=200)",
    ]
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
        "Advantage scales with task complexity:\nOn Parallel-Multiple, Pure RL (76.5%)\noutperforms Pure SFT (41.5%) by +35.0%",
        xy=(3 + w1 / 2, 76.5),
        xytext=(1.15, 88.0),
        fontsize=9.0,
        color=C_RL_DARK,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#ecfdf5", edgecolor="#6ee7b7"),
        arrowprops=dict(arrowstyle="->", color=C_RL, lw=1.4),
    )

    ax2.set_title("(b) BFCL-v3 Out-of-Domain Breakdown by Difficulty", fontsize=11.5, fontweight="bold", pad=12)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(cats_r, fontsize=10, fontweight="bold")
    ax2.set_ylabel("Strict Exact Match (%)", fontsize=10.5)
    ax2.set_ylim(0, 105)
    ax2.legend(loc="upper left", fontsize=9.5)

    fig.suptitle(
        "In-Domain Label Quality Audit (RLLA-4k) & Out-of-Domain Complexity Scaling (BFCL-v3)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = ASSETS_DIR / "fig3_deep_dive_clean_dirty_and_bfcl.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def fig4_reward_v2_staircase() -> Path:
    """Figure 4: Stepwise Scoring Structure of `rewards/rlla_reward_v2.py`."""
    fig, ax = plt.subplots(figsize=(11.5, 5.0), facecolor="#ffffff")
    style_ax(ax)

    stages = [
        "Stage 0\nMalformed JSON\nor Wrong Mode",
        "Stage 1\nValid XML Tags\n<think>+<tool_call>",
        "Stage 2\nTool Name Match\n+ Required Keys",
        "Stage 3\nCorrect Values\n(w/ Extra Keys)",
        "Stage 4\nNon-Exact Cap\n(Minor Type Diff)",
        "Stage 5\nStrict Exact Match\n(Exact Match = 1.0)",
    ]
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
        "Anti-Reward-Hacking 1 (Extra-Key Penalty):\nDeducts -0.75 per redundant parameter key\nnot present in Ground Truth",
        xy=(3, 2.38),
        xytext=(1.85, 3.55),
        fontsize=9.2,
        color="#92400e",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fffbeb", edgecolor="#fcd34d"),
        arrowprops=dict(arrowstyle="->", color=C_WARN, lw=1.4),
    )

    ax.annotate(
        "Anti-Reward-Hacking 2 (Non-Exact Cap):\nCaps partial matches at +3.4 to preserve\na +0.60 margin for Strict Exact Match (+4.0)",
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
    ax.set_title(
        "Dense Reward Shaping & Anti-Reward-Hacking Design in rlla_reward_v2.py",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )

    fig.tight_layout()
    out = ASSETS_DIR / "fig4_reward_v2_staircase.png"
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
