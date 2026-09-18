#!/usr/bin/env bash
set -euo pipefail
RUN036="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
export UV_CACHE_DIR=/workspace/run036-uv-cache UV_HTTP_TIMEOUT=120
mkdir -p "$RUN036/runtime"
python3 -m venv "$RUN036/runtime/bootstrap"
"$RUN036/runtime/bootstrap/bin/pip" install uv==0.8.15
export PATH="$RUN036/runtime/bootstrap/bin:$PATH"
uv venv --python python3 "$RUN036/runtime/venv"
uv pip install --python "$RUN036/runtime/venv/bin/python" --index-url https://pypi.org/simple --find-links https://download.pytorch.org/whl/cu128/torch/ -r "$RUN036/provenance/pip-freeze.txt"
uv pip freeze --python "$RUN036/runtime/venv/bin/python" > "$RUN036/runtime/pip-freeze.txt"
nvcc --version > "$RUN036/runtime/nvcc.txt"
nvidia-smi -q > "$RUN036/runtime/nvidia-smi-initial.txt"
"$RUN036/runtime/venv/bin/python" "$RUN036/01_prepare.py" verify
touch "$RUN036/runtime/setup-complete"
