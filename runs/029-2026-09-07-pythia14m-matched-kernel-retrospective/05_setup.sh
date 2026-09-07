#!/usr/bin/env bash
set -euo pipefail
RUN029="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export UV_CACHE_DIR=/workspace/run029-uv-cache
mkdir -p "$RUN029/runtime"
python3 -m venv "$RUN029/runtime/bootstrap"
"$RUN029/runtime/bootstrap/bin/pip" install uv==0.8.15
export PATH="$RUN029/runtime/bootstrap/bin:$PATH"
uv venv --python python3 "$RUN029/runtime/venv"
uv pip install --python "$RUN029/runtime/venv/bin/python" --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0
uv pip install --python "$RUN029/runtime/venv/bin/python" transformers==5.12.1 numpy==2.5.0 PyYAML==6.0.2 safetensors==0.8.0 ninja==1.11.1.4 setuptools==80.9.0 pip==25.2 pytest==9.0.2
uv pip freeze --python "$RUN029/runtime/venv/bin/python" > "$RUN029/runtime/pip-freeze.txt"
nvcc --version > "$RUN029/runtime/nvcc.txt"
nvidia-smi -q > "$RUN029/runtime/nvidia-smi-initial.txt"
"$RUN029/runtime/venv/bin/python" "$RUN029/04_bundle.py" verify
printf 'SETUP_COMPLETE\n'
