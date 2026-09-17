#!/usr/bin/env bash
set -euo pipefail
cd /workspace/run033
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export UV_CACHE_DIR=/workspace/run033-uv-cache
runtime/bootstrap/bin/uv pip install --python runtime/venv/bin/python --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128 -r provenance/pip-freeze.txt
runtime/bootstrap/bin/uv pip freeze --python runtime/venv/bin/python > runtime/pip-freeze.txt
nvcc --version > runtime/nvcc.txt
nvidia-smi -q > runtime/nvidia-smi-initial.txt
touch runtime/environment-ready
