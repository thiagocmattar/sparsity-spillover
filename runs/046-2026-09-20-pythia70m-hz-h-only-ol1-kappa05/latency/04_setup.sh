#!/usr/bin/env bash
set -euo pipefail
RUN046="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export UV_CACHE_DIR=/workspace/run046-latency-uv-cache
mkdir -p "$RUN046/runtime"
if test ! -e "$RUN046/runtime/environment-ready"; then
python3 -m venv "$RUN046/runtime/bootstrap"
"$RUN046/runtime/bootstrap/bin/pip" install uv==0.8.15
export PATH="$RUN046/runtime/bootstrap/bin:$PATH"
uv venv --python python3 "$RUN046/runtime/venv"
uv pip install --python "$RUN046/runtime/venv/bin/python" --index-url https://pypi.org/simple --find-links https://download.pytorch.org/whl/cu128/torch/ -r "$RUN046/provenance/pip-freeze.txt"
uv pip freeze --python "$RUN046/runtime/venv/bin/python" > "$RUN046/runtime/pip-freeze.txt"
nvcc --version > "$RUN046/runtime/nvcc.txt"
nvidia-smi -q > "$RUN046/runtime/nvidia-smi-initial.txt"
touch "$RUN046/runtime/environment-ready"
fi
if test "${1:-}" = --environment-only; then exit 0; fi
"$RUN046/runtime/venv/bin/python" "$RUN046/01_prepare.py" verify
touch "$RUN046/runtime/setup-complete"
