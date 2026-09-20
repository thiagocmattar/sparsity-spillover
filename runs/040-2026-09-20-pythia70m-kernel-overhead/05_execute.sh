#!/usr/bin/env bash
set -euo pipefail
RUN040="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$RUN040"
export PATH="$RUN040/runtime/venv/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda TORCH_EXTENSIONS_DIR="$RUN040/runtime/extensions" MAX_JOBS=2
if [ "$1" = operators ]; then
  python -u 10_test_operators.py
  exec python -u 06_control_checks.py
else
  exec python -u 03_execute.py --phase "$1" --deadline-epoch "$2" --tag "${3:-001}"
fi
