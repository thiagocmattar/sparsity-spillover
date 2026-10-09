#!/usr/bin/env bash
set -euo pipefail
cd /workspace/run069
python -m venv /workspace/run069-env
source /workspace/run069-env/bin/activate
python -m pip install --timeout 300 --retries 8 'torch==2.11.0' --index-url https://download.pytorch.org/whl/cu128
python -m pip install --timeout 300 --retries 8 'transformers==5.12.1' 'numpy==2.5.0' 'safetensors==0.8.0' 'pyyaml==6.0.3' 'ninja==1.11.1.4'
python -m pip freeze > /workspace/run069-environment.txt
export PYTHONPATH=/workspace/run069/src
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export NCU="$(command -v ncu || true)"
if [ -z "$NCU" ]; then
    NCU=$(find /opt/nvidia /usr/local -type f -name ncu -print -quit 2>/dev/null || true)
fi
printf '%s\n' "$NCU" > /workspace/run069-ncu-path.txt
