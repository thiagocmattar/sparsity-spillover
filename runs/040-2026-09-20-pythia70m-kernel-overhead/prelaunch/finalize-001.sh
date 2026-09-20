#!/usr/bin/env bash
set -euo pipefail
cd /workspace/run040
export PATH="/workspace/run040/runtime/venv/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda TORCH_EXTENSIONS_DIR=/workspace/run040/runtime/extensions MAX_JOBS=2
bash 05_execute.sh optimized 1789917919.103 001 >runtime/optimized-001.log 2>&1
echo 0 >runtime/optimized-001.exit
python -u candidates/profile_selected.py >runtime/profile-selected-001.log 2>&1
echo 0 >runtime/profile-selected-001.exit
python -u 07_reduce.py >runtime/reduce-final-001.log 2>&1
mkdir -p transfer
python -u 08_collect.py --tag 001 >transfer/collect-001.log 2>&1
