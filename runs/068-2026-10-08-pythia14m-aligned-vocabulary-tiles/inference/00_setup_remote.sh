#!/usr/bin/env bash
set -euo pipefail
cd /workspace/run068-head
python -m venv /workspace/run068-head-env
source /workspace/run068-head-env/bin/activate
python -m pip install --timeout 300 --retries 8 'torch==2.11.0' --index-url https://download.pytorch.org/whl/cu128
python -m pip install --timeout 300 --retries 8 'transformers==5.12.1' 'numpy==2.5.0' 'safetensors==0.8.0' 'pyyaml==6.0.3' 'ninja==1.11.1.4'
export PYTHONPATH=/workspace/run068-head/src
python - <<'PY'
import torch, transformers, numpy
print({'torch':torch.__version__,'transformers':transformers.__version__,'numpy':numpy.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name()})
assert torch.__version__.split('+')[0]=='2.11.0' and torch.version.cuda=='12.8'
assert transformers.__version__=='5.12.1' and numpy.__version__=='2.5.0'
PY
python -m pip freeze > /workspace/run068-head-environment.txt
