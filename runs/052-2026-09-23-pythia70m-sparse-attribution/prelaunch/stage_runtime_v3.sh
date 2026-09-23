#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="/workspace/run052/052-2026-09-23-pythia70m-sparse-attribution"
BASE_DIR="$(dirname "$TASK_DIR")/049-2026-09-22-pythia70m-short-row-limits"
export PATH="$BASE_DIR/runtime/venv/bin:/usr/local/cuda/bin:$PATH" CUDA_HOME=/usr/local/cuda
export RUN052_BASE="$BASE_DIR" TORCH_EXTENSIONS_DIR="$BASE_DIR/runtime/extensions"
export TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=4
unset CUBLAS_WORKSPACE_CONFIG
exec "$BASE_DIR/runtime/venv/bin/python" -u "$TASK_DIR/18_execute_v3.py" "$@"
