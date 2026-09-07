#!/usr/bin/env bash
set -euo pipefail
RUN029="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$RUN029/runtime/venv/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export TORCH_EXTENSIONS_DIR="$RUN029/runtime/extensions"
export TRITON_CACHE_DIR="$RUN029/runtime/triton"
export MAX_JOBS=2
export HF_HUB_OFFLINE=1
unset CUBLAS_WORKSPACE_CONFIG
exec python -u "$RUN029/03_matrix.py" --phase "$1" --deadline "$2"
