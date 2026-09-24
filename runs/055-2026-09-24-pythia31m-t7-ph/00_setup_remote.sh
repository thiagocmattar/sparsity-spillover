#!/usr/bin/env bash
set -euo pipefail
repo=/workspace/sparsity-spillover
export UV_CACHE_DIR=/workspace/run055-uv-cache UV_HTTP_TIMEOUT=120
mkdir -p /workspace/run055-control
python3 -m venv /workspace/run055-bootstrap
/workspace/run055-bootstrap/bin/pip install uv==0.8.15
uv=/workspace/run055-bootstrap/bin/uv
"$uv" venv --python python3 /workspace/run055-venv
"$uv" pip install --python /workspace/run055-venv/bin/python --index-url https://pypi.org/simple --find-links https://download.pytorch.org/whl/cu128/torch/ torch==2.11.0+cu128 datasets==5.0.0 matplotlib==3.11.0 numpy==2.5.0 pyyaml==6.0.3 safetensors==0.8.0 transformers==5.12.1 ninja==1.13.0 pytest==9.0.2
"$uv" pip install --python /workspace/run055-venv/bin/python --no-deps -e "$repo"
"$uv" pip freeze --python /workspace/run055-venv/bin/python > /workspace/run055-control/pip-freeze.txt
/workspace/run055-venv/bin/python -c 'import sys,torch,transformers;assert sys.version_info[:2]==(3,12);assert torch.__version__=="2.11.0+cu128";assert transformers.__version__=="5.12.1";assert torch.version.cuda=="12.8"'
touch /workspace/run055-control/environment-ready
