#!/usr/bin/env bash
set -euo pipefail
RUN041="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$RUN041/runtime/venv/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export TORCH_EXTENSIONS_DIR="$RUN041/runtime/extensions"
export TRITON_CACHE_DIR="$RUN041/runtime/triton"
export MAX_JOBS=2
export HF_HUB_OFFLINE=1
unset CUBLAS_WORKSPACE_CONFIG
exec python -u "$RUN041/03_execute.py" --phase "$1" --deadline-epoch "$2" --tag "${3:-001}"
