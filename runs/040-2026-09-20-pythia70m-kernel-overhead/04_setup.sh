#!/usr/bin/env bash
set -euo pipefail
RUN040="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
export UV_CACHE_DIR=/workspace/run040-uv-cache UV_HTTP_TIMEOUT=120
mkdir -p "$RUN040/runtime"
python3 -m venv "$RUN040/runtime/bootstrap"
"$RUN040/runtime/bootstrap/bin/pip" install uv==0.8.15
export PATH="$RUN040/runtime/bootstrap/bin:$PATH"
uv venv --python python3 "$RUN040/runtime/venv"
uv pip install --python "$RUN040/runtime/venv/bin/python" --index-url https://pypi.org/simple --find-links https://download.pytorch.org/whl/cu128/torch/ -r "$RUN040/provenance/pip-freeze.txt"
uv pip freeze --python "$RUN040/runtime/venv/bin/python" > "$RUN040/runtime/pip-freeze.txt"
nvcc --version > "$RUN040/runtime/nvcc.txt"
nvidia-smi -q > "$RUN040/runtime/nvidia-smi-initial.txt"
"$RUN040/runtime/venv/bin/python" "$RUN040/01_prepare.py" verify
touch "$RUN040/runtime/setup-complete"
