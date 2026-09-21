#!/usr/bin/env bash
set -euo pipefail
TASK_DIR=/workspace/run048
export PATH=/tmp/run048-local-runtime/bootstrap/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
export UV_CACHE_DIR=/tmp/run048-uv-cache UV_HTTP_TIMEOUT=120
python3 /workspace/run048-control/reuse_image_packages_003.py
uv pip install --python /tmp/run048-local-runtime/venv/bin/python --index-url https://pypi.org/simple --find-links https://download.pytorch.org/whl/cu128/torch/ -r "$TASK_DIR/provenance/pip-freeze.txt"
ln -s /tmp/run048-local-runtime/venv "$TASK_DIR/runtime/venv"
mkdir -p /tmp/run048-local-runtime/extensions
ln -s /tmp/run048-local-runtime/extensions "$TASK_DIR/runtime/extensions"
uv pip freeze --python "$TASK_DIR/runtime/venv/bin/python" > "$TASK_DIR/runtime/pip-freeze.txt"
nvcc --version > "$TASK_DIR/runtime/nvcc.txt"
nvidia-smi -q > "$TASK_DIR/runtime/nvidia-smi-initial.txt"
touch "$TASK_DIR/runtime/environment-ready"
