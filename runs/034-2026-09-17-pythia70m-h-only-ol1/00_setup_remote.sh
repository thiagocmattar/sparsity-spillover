#!/usr/bin/env bash
set -euo pipefail
repo=/workspace/sparsity-spillover
run=$repo/runs/034-2026-09-17-pythia70m-h-only-ol1
mkdir -p /workspace/run034-control
python3.12 -m venv /opt/run034-bootstrap
/opt/run034-bootstrap/bin/pip install -q uv==0.6.17
uv=/opt/run034-bootstrap/bin/uv
$uv venv --python python3.12 /opt/run034-venv
export UV_HTTP_TIMEOUT=120
export UV_CACHE_DIR=/opt/run034-uv-cache
$uv pip install --python /opt/run034-venv/bin/python --index-url https://pypi.org/simple --find-links https://download.pytorch.org/whl/cu128/torch/ -r "$run/prelaunch/pip-freeze-source.txt"
$uv pip install --python /opt/run034-venv/bin/python --no-deps -e "$repo"
$uv pip check --python /opt/run034-venv/bin/python
$uv pip freeze --python /opt/run034-venv/bin/python > /workspace/run034-control/pip-freeze.txt
/opt/run034-venv/bin/python -c 'import torch,transformers;assert torch.__version__.split("+")[0]=="2.11.0";assert torch.version.cuda=="12.8";assert transformers.__version__=="5.12.1"'
touch /workspace/run034-control/environment-ready
