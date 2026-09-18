#!/usr/bin/env bash
set -euo pipefail
cd /workspace/run036
export PATH=/workspace/run036/runtime/venv/bin:/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda TORCH_EXTENSIONS_DIR=/workspace/run036/runtime/extensions MAX_JOBS=2
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
unset CUBLAS_WORKSPACE_CONFIG
python -u 03_execute.py --phase preflight --deadline-epoch "$1"
python -u 03_execute.py --phase scientific --deadline-epoch "$1"
python -u 08_collect.py
