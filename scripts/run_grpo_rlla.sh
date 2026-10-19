#!/usr/bin/env bash
# =============================================================================
# Stage 1 Baseline: ToolRL (RLLA-4k) GRPO Training with VeRL
# Supports both Single-Node (8 GPUs) and Multi-Node Ray Cluster (4x8=32 GPUs)
# =============================================================================
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VERL_ROOT="${VERL_ROOT:-/home/ma-user/work/LB/verl}"
PYTHON_BIN="${PYTHON_BIN:-/home/ma-user/anaconda3/envs/verl_train/bin/python}"

# ----------------------- Cluster & Model Configuration -----------------------
NNODES="${NNODES:-4}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
HEAD_ADDR="${HEAD_ADDR:-172.18.39.173}"
NET_IFNAME="${NET_IFNAME:-ens2f5}"
export RAY_ADDRESS="${RAY_ADDRESS:-${HEAD_ADDR}:6379}"
export PYTHONPATH="${VERL_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"

# Base model choices:
#   - ${PROJECT_ROOT}/models/Qwen3-4B-Base-rlla (Zero-RL patched tokenizer for Qwen3-4B-Base)
#   - /home/ma-user/work/models/Qwen3-0.6B
#   - /home/ma-user/work/models/Qwen3-30B-A3B-Instruct-2507
DEFAULT_MODEL_PATH="${PROJECT_ROOT}/models/Qwen3-4B-Base-rlla"
if [[ ! -f "${DEFAULT_MODEL_PATH}/tokenizer_config.json" ]]; then
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/prepare_qwen3_base_tokenizer.py"
fi
MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
MODEL_TAG="$(basename "${MODEL_PATH}")"

PROJECT_NAME="${PROJECT_NAME:-agent-rl-lab}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-grpo-rlla4k-${MODEL_TAG}-n${NNODES}g${GPUS_PER_NODE}}"

TRAIN_FILE="${TRAIN_FILE:-${PROJECT_ROOT}/data/rlla_4k/train.parquet}"
TEST_FILE="${TEST_FILE:-${PROJECT_ROOT}/data/rlla_4k/test.parquet}"
REWARD_FN_PATH="${REWARD_FN_PATH:-${PROJECT_ROOT}/rewards/rlla_reward.py}"
CKPTS_DIR="${CKPTS_DIR:-${PROJECT_ROOT}/outputs/ckpts/${EXPERIMENT_NAME}}"

GEN_TP="${GEN_TP:-2}"
ROLLOUT_N="${ROLLOUT_N:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"

LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
METRICS_DIR="${LOG_DIR}/metrics"
TB_DIR="${LOG_DIR}/tensorboard/${EXPERIMENT_NAME}"
VAL_DUMP_DIR="${LOG_DIR}/val_dumps/${EXPERIMENT_NAME}"

mkdir -p "${LOG_DIR}" "${METRICS_DIR}" "${TB_DIR}" "${VAL_DUMP_DIR}" "${CKPTS_DIR}"
START_TIME="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${EXPERIMENT_NAME}_${START_TIME}.log"
METRICS_FILE="${METRICS_DIR}/${EXPERIMENT_NAME}_${START_TIME}.jsonl"
ln -sfn "${LOG_FILE}" "${LOG_DIR}/latest.log"

NET_ARGS=(
    "+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=${VERL_ROOT}:${PROJECT_ROOT}"
    "+ray_kwargs.ray_init.runtime_env.env_vars.GLOO_SOCKET_IFNAME=${NET_IFNAME}"
    "+ray_kwargs.ray_init.runtime_env.env_vars.NCCL_SOCKET_IFNAME=${NET_IFNAME}"
    "+ray_kwargs.ray_init.runtime_env.env_vars.TP_SOCKET_IFNAME=${NET_IFNAME}"
    "+ray_kwargs.ray_init.runtime_env.env_vars.TENSORBOARD_DIR=${TB_DIR}"
    "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_PATH=${METRICS_FILE}"
)

DATA_ARGS=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.train_batch_size="${TRAIN_BATCH_SIZE}"
    data.max_prompt_length=4096
    data.max_response_length=1024
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.shuffle=True
)

MODEL_ARGS=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.trust_remote_code=True
)

ACTOR_ARGS=(
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}"
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0.0
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
)

REF_ARGS=(
    actor_rollout_ref.ref.strategy=fsdp2
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.ref.fsdp_config.param_offload=False
)

ROLLOUT_ARGS=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size="${GEN_TP}"
    actor_rollout_ref.rollout.n="${ROLLOUT_N}"
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5
    actor_rollout_ref.rollout.max_model_len=6144
    actor_rollout_ref.rollout.max_num_batched_tokens=8192
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.free_cache_engine=True
)

REWARD_ARGS=(
    reward.reward_manager.name=naive
    reward.custom_reward_function.path="${REWARD_FN_PATH}"
    reward.custom_reward_function.name=compute_score
    custom_reward_function.path="${REWARD_FN_PATH}"
    custom_reward_function.name=compute_score
)

TRAINER_ARGS=(
    trainer.critic_warmup=0
    trainer.logger="['console','tensorboard','file']"
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.n_gpus_per_node="${GPUS_PER_NODE}"
    trainer.nnodes="${NNODES}"
    trainer.default_local_dir="${CKPTS_DIR}"
    trainer.validation_data_dir="${VAL_DUMP_DIR}"
    trainer.val_before_train=True
    trainer.save_freq=20
    trainer.test_freq=10
    trainer.total_epochs=5
)

"${PYTHON_BIN}" -m verl.trainer.main_ppo \
    "${DATA_ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
    "${ACTOR_ARGS[@]}" \
    "${REF_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${REWARD_ARGS[@]}" \
    "${NET_ARGS[@]}" \
    "${TRAINER_ARGS[@]}" \
    "$@" 2>&1 | tee "${LOG_FILE}"
