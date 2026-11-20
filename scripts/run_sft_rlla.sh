#!/usr/bin/env bash
# =============================================================================
# Stage 1 SFT Baseline: ToolRL (RLLA-4k) Full-Parameter SFT with LLaMA-Factory
# Default: Single-Node 8x A800-80GB with DeepSpeed ZeRO-2 (~2.5 mins for 2 epochs)
# =============================================================================
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LF_ROOT="${LF_ROOT:-/home/ma-user/work/LB/LlamaFactory}"
LF_ENV_DIR="${LF_ENV_DIR:-/home/ma-user/anaconda3/envs/llamafactory_fix1}"
PYTHON_BIN="${PYTHON_BIN:-${LF_ENV_DIR}/bin/python}"
LF_CLI="${LF_CLI:-${LF_ENV_DIR}/bin/llamafactory-cli}"

export PATH="${LF_ENV_DIR}/bin:${PATH}"
export PYTHONPATH="${LF_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"

# ----------------------- Model & Data Preparation ----------------------------
DEFAULT_MODEL_PATH="${PROJECT_ROOT}/models/Qwen3-4B-Base-rlla"
if [[ ! -f "${DEFAULT_MODEL_PATH}/tokenizer_config.json" ]]; then
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/prepare_qwen3_base_tokenizer.py"
fi

if [[ ! -f "${PROJECT_ROOT}/data/rlla_4k/sft_train.json" || ! -f "${PROJECT_ROOT}/data/rlla_4k/dataset_info.json" ]]; then
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/prepare_sft_data.py"
fi

# ----------------------- Training Hyperparameters ----------------------------
MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
MODEL_TAG="$(basename "${MODEL_PATH}")"
NUM_EPOCHS="${NUM_EPOCHS:-2.0}"
LR="${LR:-1.0e-5}"
PER_DEVICE_BS="${PER_DEVICE_BS:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"

EPOCH_TAG="ep${NUM_EPOCHS%.0}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-sft-rlla4k-${MODEL_TAG}-${EPOCH_TAG}}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/sft/${EXPERIMENT_NAME}}"
TB_DIR="${TB_DIR:-${PROJECT_ROOT}/logs/tensorboard/${EXPERIMENT_NAME}}"
CONFIG_FILE="${CONFIG_FILE:-${PROJECT_ROOT}/configs/sft_qwen3_4b_base_rlla.yaml}"

LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}" "${TB_DIR}"
START_TIME="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${EXPERIMENT_NAME}_${START_TIME}.log"
ln -sfn "${LOG_FILE}" "${LOG_DIR}/latest_sft.log"

export TENSORBOARD_LOGGING_DIR="${TB_DIR}"
export FORCE_TORCHRUN="${FORCE_TORCHRUN:-1}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29505}"

"${LF_CLI}" train "${CONFIG_FILE}" \
    model_name_or_path="${MODEL_PATH}" \
    output_dir="${OUTPUT_DIR}" \
    num_train_epochs="${NUM_EPOCHS}" \
    learning_rate="${LR}" \
    per_device_train_batch_size="${PER_DEVICE_BS}" \
    gradient_accumulation_steps="${GRAD_ACCUM}" \
    "$@" 2>&1 | tee "${LOG_FILE}"
