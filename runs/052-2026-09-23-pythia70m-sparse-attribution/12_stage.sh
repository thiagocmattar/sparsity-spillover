#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$TASK_DIR")/049-2026-09-22-pythia70m-short-row-limits"
export PATH="/usr/local/cuda/bin:$PATH" CUDA_HOME=/usr/local/cuda
export RUN052_BASE="$BASE_DIR" TORCH_EXTENSIONS_DIR="$BASE_DIR/runtime/extensions"
export TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=4
unset CUBLAS_WORKSPACE_CONFIG
exec "$BASE_DIR/runtime/venv/bin/python" -u "$TASK_DIR/09_execute.py" "$@"
