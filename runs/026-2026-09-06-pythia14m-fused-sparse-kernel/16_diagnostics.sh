#!/usr/bin/env bash
set -euo pipefail
cd /workspace/sparsity-spillover
RUN=runs/026-2026-09-06-pythia14m-fused-sparse-kernel
export PATH="$PWD/$RUN/runtime/venv/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
unset CUBLAS_WORKSPACE_CONFIG
python "$RUN/autoresearch/diagnostics.py"
