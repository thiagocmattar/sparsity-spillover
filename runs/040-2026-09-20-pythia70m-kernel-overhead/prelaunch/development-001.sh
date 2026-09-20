#!/usr/bin/env bash
set -euo pipefail
cd /workspace/run040
while [ ! -e runtime/diagnostic-complete ]; do
  if [ -e runtime/pipeline-001.exit ]; then exit 1; fi
  sleep 10
done
export PATH="/workspace/run040/runtime/venv/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda TORCH_EXTENSIONS_DIR=/workspace/run040/runtime/extensions MAX_JOBS=2
python -u 07_reduce.py > runtime/reduce-diagnostic-001.log 2>&1
python -u candidates/develop.py 1789917919.103 > runtime/development-001.log 2>&1
