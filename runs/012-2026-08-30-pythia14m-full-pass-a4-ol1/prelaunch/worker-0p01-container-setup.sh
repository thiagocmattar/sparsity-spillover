#!/usr/bin/env bash
set -euo pipefail

repo=/workspace/sparsity-spillover
environment=/root/run012-venv
packet=/workspace/run012-preflight-packet

test -d "$repo/.git"
mkdir -p "$packet"
python3.12 -m venv "$environment"
"$environment/bin/python" -m pip install --no-cache-dir --upgrade \
  pip==25.0.1 setuptools==70.2.0
"$environment/bin/python" -m pip install --no-cache-dir \
  torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
"$environment/bin/python" -m pip install --no-cache-dir \
  datasets==5.0.0 \
  matplotlib==3.11.0 \
  numpy==2.5.0 \
  pyyaml==6.0.3 \
  safetensors==0.8.0 \
  transformers==5.12.1
"$environment/bin/python" -m pip install --no-cache-dir --no-deps -e "$repo"
"$environment/bin/python" -m pip check
"$environment/bin/python" -m pip freeze > "$packet/runpod-pip-freeze.txt"
"$environment/bin/python" -c 'import torch, transformers; assert torch.__version__.split("+", 1)[0] == "2.11.0"; assert torch.version.cuda == "12.8"; assert transformers.__version__ == "5.12.1"'
