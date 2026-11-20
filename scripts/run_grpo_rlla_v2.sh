#!/usr/bin/env bash
# =============================================================================
# Stage 1 Upgraded Pure RL (v2 Reward + +5 Epochs Continuation = 9 Epochs Total)
#
# Key Improvements in `rewards/rlla_reward_v2.py`:
#   1. Penalizes extra/redundant optional parameter keys (-0.75 per extra key)
#      and caps non-exact tool calls at +2.4/3.0 so strict Exact Match (+3.0/3.0)
#      always has a clear +0.60 advantage margin.
#   2. Two-tier value matching (+1.0 for exact type & value, +0.6 for int/str
#      normalized match) to eliminate false-negative gradient noise.
#   3. Balanced No-Tool / Clarification reward (+3.5 total instead of +1.0) so
#      the policy properly abstains when required parameters are missing.
# =============================================================================
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Default starting model: Pure RL Epoch 4 (Step 120) merged checkpoint
# (Already mastered 97.5% strict XML format & 99.8% genuine CoT, saving 2.3 epochs of format warmup)
DEFAULT_CONT_MODEL="${PROJECT_ROOT}/outputs/merged/grpo-rlla4k-Qwen3-4B-Base-rlla-step120"

export MODEL_PATH="${MODEL_PATH:-${DEFAULT_CONT_MODEL}}"
export REWARD_FN_PATH="${REWARD_FN_PATH:-${PROJECT_ROOT}/rewards/rlla_reward_v2.py}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-grpo-rlla4k-v2reward-PureRL-cont5ep-n4g8}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-5}"

exec bash "${SCRIPT_DIR}/run_grpo_rlla.sh" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    actor_rollout_ref.actor.kl_loss_coef=0.0005 \
    "$@"
