#!/usr/bin/env bash
set -euo pipefail
repo=/workspace/sparsity-spillover
run="$repo/runs/046-2026-09-20-pythia70m-hz-h-only-ol1-kappa05"
export UV_CACHE_DIR=/workspace/run046-uv-cache UV_HTTP_TIMEOUT=120
mkdir -p /workspace/run046-control
python3 -m venv /workspace/run046-bootstrap
/workspace/run046-bootstrap/bin/pip install uv==0.8.15
uv=/workspace/run046-bootstrap/bin/uv
"$uv" venv --python python3 /workspace/run046-venv
"$uv" pip install --python /workspace/run046-venv/bin/python --index-url https://pypi.org/simple --find-links https://download.pytorch.org/whl/cu128/torch/ torch==2.11.0+cu128 datasets==5.0.0 matplotlib==3.11.0 numpy==2.5.0 pyyaml==6.0.3 safetensors==0.8.0 transformers==5.12.1
"$uv" pip install --python /workspace/run046-venv/bin/python --no-deps -e "$repo"
"$uv" pip freeze --python /workspace/run046-venv/bin/python > /workspace/run046-control/pip-freeze.txt
/workspace/run046-venv/bin/python -c 'import torch,transformers;assert torch.__version__=="2.11.0+cu128";assert transformers.__version__=="5.12.1"'
touch /workspace/run046-control/environment-ready
