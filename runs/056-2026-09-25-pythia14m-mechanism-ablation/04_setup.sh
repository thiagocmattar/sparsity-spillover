#!/usr/bin/env bash
set -euo pipefail
RUN056="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export UV_HTTP_TIMEOUT=120
export UV_CACHE_DIR=/opt/run056-uv-cache
mkdir -p "$RUN056/runtime"
mkdir -p /opt/run056
python3 -m venv /opt/run056/bootstrap
/opt/run056/bootstrap/bin/pip install uv==0.8.15
UV=/opt/run056/bootstrap/bin/uv
"$UV" venv --python python3 /opt/run056/venv
"$UV" pip install --python /opt/run056/venv/bin/python --index-url https://pypi.org/simple \
  --find-links https://download.pytorch.org/whl/cu128/torch/ -r "$RUN056/provenance/pip-freeze.txt"
"$UV" pip freeze --python /opt/run056/venv/bin/python > "$RUN056/runtime/pip-freeze.txt"
nvcc --version > "$RUN056/runtime/nvcc.txt"
nvidia-smi -q > "$RUN056/runtime/nvidia-smi-initial.txt"
/opt/run056/venv/bin/python "$RUN056/01_prepare.py" verify
touch "$RUN056/runtime/setup-complete"
