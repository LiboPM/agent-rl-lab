#!/usr/bin/env bash
# Download ToolRL (RLLA-4k) dataset in VeRL Parquet format via hf-mirror.com
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${PROJECT_ROOT}/data/rlla_4k"
PROXY="${HTTPS_PROXY:-http://proxy-notebook.modelarts.com:8083}"

mkdir -p "${DATA_DIR}"

echo "Downloading ToolRL rlla_4k train.parquet (3,920 samples)..."
curl -L -x "${PROXY}" \
  "https://hf-mirror.com/datasets/RamAnanth1/toolrl-rlla4k/resolve/main/data/train-00000-of-00001.parquet" \
  -o "${DATA_DIR}/train.parquet"

echo "Downloading ToolRL rlla_4k test.parquet (80 samples)..."
curl -L -x "${PROXY}" \
  "https://hf-mirror.com/datasets/RamAnanth1/toolrl-rlla4k/resolve/main/data/test-00000-of-00001.parquet" \
  -o "${DATA_DIR}/test.parquet"

echo "Done! Files saved to ${DATA_DIR}:"
ls -lh "${DATA_DIR}"
