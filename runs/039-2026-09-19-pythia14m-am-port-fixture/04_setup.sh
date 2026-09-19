#!/usr/bin/env bash
set -euo pipefail
RUN038="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH=/opt/run038-venv/bin:/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export UV_HTTP_TIMEOUT=120
export UV_CACHE_DIR=/opt/run038-uv-cache
mkdir -p "$RUN038/runtime"
python3 -m venv /opt/run038-bootstrap
/opt/run038-bootstrap/bin/pip install uv==0.8.15
UV=/opt/run038-bootstrap/bin/uv
"$UV" venv --python python3 /opt/run038-venv
"$UV" pip install --python /opt/run038-venv/bin/python --index-url https://pypi.org/simple \
  --find-links https://download.pytorch.org/whl/cu128/torch/ -r "$RUN038/provenance/pip-freeze.txt"
"$UV" pip freeze --python /opt/run038-venv/bin/python > "$RUN038/runtime/pip-freeze.txt"
nvcc --version > "$RUN038/runtime/nvcc.txt"
nvidia-smi -q > "$RUN038/runtime/nvidia-smi-initial.txt"
/opt/run038-venv/bin/python "$RUN038/01_prepare.py" verify
touch "$RUN038/runtime/setup-complete"
