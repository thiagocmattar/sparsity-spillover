#!/usr/bin/env bash
set -euo pipefail
RUN045="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
export UV_CACHE_DIR=/workspace/run045-uv-cache UV_HTTP_TIMEOUT=120
mkdir -p "$RUN045/runtime"
python3 -m venv "$RUN045/runtime/bootstrap"
"$RUN045/runtime/bootstrap/bin/pip" install uv==0.8.15
export PATH="$RUN045/runtime/bootstrap/bin:$PATH"
uv venv --python python3 "$RUN045/runtime/venv"
uv pip install --python "$RUN045/runtime/venv/bin/python" --index-url https://pypi.org/simple --find-links https://download.pytorch.org/whl/cu128/torch/ -r "$RUN045/provenance/pip-freeze.txt"
uv pip freeze --python "$RUN045/runtime/venv/bin/python" > "$RUN045/runtime/pip-freeze.txt"
nvcc --version > "$RUN045/runtime/nvcc.txt"
nvidia-smi -q > "$RUN045/runtime/nvidia-smi-initial.txt"
touch "$RUN045/runtime/environment-ready"
