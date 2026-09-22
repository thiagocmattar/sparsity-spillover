#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
export UV_CACHE_DIR=/tmp/run049-uv-cache UV_HTTP_TIMEOUT=120
mkdir -p "$TASK_DIR/runtime" /tmp/run049-local-runtime
python3 -m venv /tmp/run049-local-runtime/bootstrap
/tmp/run049-local-runtime/bootstrap/bin/pip install uv==0.8.15
export PATH=/tmp/run049-local-runtime/bootstrap/bin:$PATH
uv venv --python python3 /tmp/run049-local-runtime/venv
uv pip install --python /tmp/run049-local-runtime/venv/bin/python --index-url https://pypi.org/simple --find-links https://download.pytorch.org/whl/cu128/torch/ -r "$TASK_DIR/provenance/pip-freeze.txt"
ln -s /tmp/run049-local-runtime/venv "$TASK_DIR/runtime/venv"
mkdir -p /tmp/run049-local-runtime/extensions
ln -s /tmp/run049-local-runtime/extensions "$TASK_DIR/runtime/extensions"
uv pip freeze --python "$TASK_DIR/runtime/venv/bin/python" > "$TASK_DIR/runtime/pip-freeze.txt"
nvcc --version > "$TASK_DIR/runtime/nvcc.txt"
nvidia-smi -q > "$TASK_DIR/runtime/nvidia-smi-initial.txt"
touch "$TASK_DIR/runtime/environment-ready"
