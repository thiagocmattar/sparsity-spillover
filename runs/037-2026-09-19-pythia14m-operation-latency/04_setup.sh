#!/usr/bin/env bash
set -euo pipefail
RUN037="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export UV_HTTP_TIMEOUT=120
export UV_CACHE_DIR=/workspace/run037-uv-cache
mkdir -p "$RUN037/runtime"
python3 -m venv "$RUN037/runtime/bootstrap"
"$RUN037/runtime/bootstrap/bin/pip" install uv==0.8.15
UV="$RUN037/runtime/bootstrap/bin/uv"
"$UV" venv --python python3 "$RUN037/runtime/venv"
"$UV" pip install --python "$RUN037/runtime/venv/bin/python" --index-url https://pypi.org/simple \
  --find-links https://download.pytorch.org/whl/cu128/torch/ -r "$RUN037/provenance/pip-freeze.txt"
"$UV" pip freeze --python "$RUN037/runtime/venv/bin/python" > "$RUN037/runtime/pip-freeze.txt"
nvcc --version > "$RUN037/runtime/nvcc.txt"
nvidia-smi -q > "$RUN037/runtime/nvidia-smi-initial.txt"
"$RUN037/runtime/venv/bin/python" "$RUN037/01_prepare.py" verify
touch "$RUN037/runtime/setup-complete"
