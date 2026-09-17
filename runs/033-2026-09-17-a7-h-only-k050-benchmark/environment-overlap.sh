#!/usr/bin/env bash
set -euo pipefail
RUN033="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export UV_CACHE_DIR=/workspace/run033-uv-cache
mkdir -p "$RUN033/runtime"
python3 -m venv "$RUN033/runtime/bootstrap"
"$RUN033/runtime/bootstrap/bin/pip" install uv==0.8.15
export PATH="$RUN033/runtime/bootstrap/bin:$PATH"
uv venv --python python3 "$RUN033/runtime/venv"
uv pip install --python "$RUN033/runtime/venv/bin/python" --extra-index-url https://download.pytorch.org/whl/cu128 -r "$RUN033/provenance/pip-freeze.txt"
uv pip freeze --python "$RUN033/runtime/venv/bin/python" > "$RUN033/runtime/pip-freeze.txt"
nvcc --version > "$RUN033/runtime/nvcc.txt"
nvidia-smi -q > "$RUN033/runtime/nvidia-smi-initial.txt"
touch "$RUN033/runtime/environment-ready"
