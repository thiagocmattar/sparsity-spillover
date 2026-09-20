#!/usr/bin/env bash
set -euo pipefail
RUN043="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export UV_CACHE_DIR=/workspace/run043-latency-uv-cache
mkdir -p "$RUN043/runtime"
if test ! -e "$RUN043/runtime/environment-ready"; then
python3 -m venv "$RUN043/runtime/bootstrap"
"$RUN043/runtime/bootstrap/bin/pip" install uv==0.8.15
export PATH="$RUN043/runtime/bootstrap/bin:$PATH"
uv venv --python python3 "$RUN043/runtime/venv"
uv pip install --python "$RUN043/runtime/venv/bin/python" --index-url https://pypi.org/simple --find-links https://download.pytorch.org/whl/cu128/torch/ -r "$RUN043/provenance/pip-freeze.txt"
uv pip freeze --python "$RUN043/runtime/venv/bin/python" > "$RUN043/runtime/pip-freeze.txt"
nvcc --version > "$RUN043/runtime/nvcc.txt"
nvidia-smi -q > "$RUN043/runtime/nvidia-smi-initial.txt"
touch "$RUN043/runtime/environment-ready"
fi
if test "${1:-}" = --environment-only; then exit 0; fi
"$RUN043/runtime/venv/bin/python" "$RUN043/01_prepare.py" verify
touch "$RUN043/runtime/setup-complete"
