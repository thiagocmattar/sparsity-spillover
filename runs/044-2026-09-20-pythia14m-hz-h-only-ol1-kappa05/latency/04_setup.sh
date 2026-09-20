#!/usr/bin/env bash
set -euo pipefail
RUN044="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export UV_CACHE_DIR=/workspace/run044-latency-uv-cache
mkdir -p "$RUN044/runtime"
if test ! -e "$RUN044/runtime/environment-ready"; then
python3 -m venv "$RUN044/runtime/bootstrap"
"$RUN044/runtime/bootstrap/bin/pip" install uv==0.8.15
export PATH="$RUN044/runtime/bootstrap/bin:$PATH"
uv venv --python python3 "$RUN044/runtime/venv"
uv pip install --python "$RUN044/runtime/venv/bin/python" --index-url https://pypi.org/simple --find-links https://download.pytorch.org/whl/cu128/torch/ -r "$RUN044/provenance/pip-freeze.txt"
uv pip freeze --python "$RUN044/runtime/venv/bin/python" > "$RUN044/runtime/pip-freeze.txt"
nvcc --version > "$RUN044/runtime/nvcc.txt"
nvidia-smi -q > "$RUN044/runtime/nvidia-smi-initial.txt"
touch "$RUN044/runtime/environment-ready"
fi
if test "${1:-}" = --environment-only; then exit 0; fi
"$RUN044/runtime/venv/bin/python" "$RUN044/01_prepare.py" verify
touch "$RUN044/runtime/setup-complete"
