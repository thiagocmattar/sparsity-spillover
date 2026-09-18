#!/usr/bin/env bash
set -euo pipefail
RUN035="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$RUN035"
export PATH="$RUN035/runtime/venv/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda TORCH_EXTENSIONS_DIR="$RUN035/runtime/extensions" MAX_JOBS=2
if [ "$1" = operators ]; then
  exec python -u 10_test_operators.py
else
  exec python -u 03_execute.py --phase "$1" --deadline-epoch "$2" --tag "${3:-001}"
fi
