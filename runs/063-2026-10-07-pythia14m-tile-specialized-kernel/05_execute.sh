#!/usr/bin/env bash
set -euo pipefail
RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH=/opt/run063-fast/venv/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
export RUN063_ARCHIVE=/opt/run063-fast/archive/root
export TORCH_EXTENSIONS_DIR=/opt/run063-fast/extensions TRITON_CACHE_DIR=/opt/run063-fast/triton
export MAX_JOBS=2 HF_HUB_OFFLINE=1
unset CUBLAS_WORKSPACE_CONFIG
exec python -u "$RUN/03_execute.py" --phase "$1" --deadline-epoch "$2" --tag "${3:-001}"
