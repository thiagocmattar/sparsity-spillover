#!/usr/bin/env bash
set -uo pipefail
RUN038="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$RUN038"
export PATH=/opt/run038-venv/bin:/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export MAX_JOBS=2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TORCH_EXTENSIONS_DIR=/opt/run038-extensions
PY=/opt/run038-venv/bin/python
test -f runtime/setup-complete || exit 2
code=0
"$PY" 01_prepare.py verify && \
"$PY" -u 06_cuda_checks.py && \
"$PY" -u 03_execute.py --phase smoke --deadline-epoch "$1" && \
"$PY" -u 03_execute.py --phase scientific --deadline-epoch "$1" && \
"$PY" 07_reduce.py || code=$?
printf '%s\n' "$code" > runtime/science.exit
"$PY" 08_collect.py || exit 3
exit "$code"
