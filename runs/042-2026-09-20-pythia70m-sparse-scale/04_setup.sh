#!/usr/bin/env bash
set -euo pipefail
RUN042="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
export UV_CACHE_DIR=/workspace/run042-uv-cache UV_HTTP_TIMEOUT=120
mkdir -p "$RUN042/runtime"
python3 -m venv "$RUN042/runtime/bootstrap"
"$RUN042/runtime/bootstrap/bin/pip" install uv==0.8.15
export PATH="$RUN042/runtime/bootstrap/bin:$PATH"
uv venv --python python3 "$RUN042/runtime/venv"
uv pip install --python "$RUN042/runtime/venv/bin/python" --index-url https://pypi.org/simple --find-links https://download.pytorch.org/whl/cu128/torch/ -r "$RUN042/provenance/pip-freeze.txt"
uv pip freeze --python "$RUN042/runtime/venv/bin/python" > "$RUN042/runtime/pip-freeze.txt"
nvcc --version > "$RUN042/runtime/nvcc.txt"
nvidia-smi -q > "$RUN042/runtime/nvidia-smi-initial.txt"
"$RUN042/runtime/venv/bin/python" "$RUN042/01_prepare.py" verify
touch "$RUN042/runtime/setup-complete"
