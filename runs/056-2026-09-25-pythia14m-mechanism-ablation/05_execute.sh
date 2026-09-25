#!/usr/bin/env bash
set -euo pipefail
RUN056="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$RUN056"
export PATH=/opt/run056/venv/bin:/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export MAX_JOBS=2
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TORCH_EXTENSIONS_DIR=/opt/run056/extensions
export PYTHONUNBUFFERED=1
PY=/opt/run056/venv/bin/python
test -f runtime/setup-complete
"$PY" 01_prepare.py verify
"$PY" -u 06_cuda_checks.py
"$PY" 14_compiler.py
"$PY" -u 03_execute.py --phase smoke --deadline-epoch "$1"
"$PY" -u 03_execute.py --phase scientific --deadline-epoch "$1"
"$PY" 07_reduce.py
# Collect only after this worker exits, so runtime logs are immutable.
