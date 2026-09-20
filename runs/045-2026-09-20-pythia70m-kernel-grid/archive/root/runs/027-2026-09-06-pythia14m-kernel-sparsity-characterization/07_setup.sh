#!/usr/bin/env bash
set -euo pipefail
run027_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$run027_dir/runtime"
export UV_CACHE_DIR="$run027_dir/runtime/uv-cache"
export PATH="/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
command -v nvcc
command -v g++
python3 -c 'import sys; assert sys.version_info[:2] == (3,12)'
if ! command -v uv >/dev/null; then python3 -m pip install --break-system-packages uv==0.8.15; fi
if [[ ! -x "$run027_dir/runtime/venv/bin/python" ]]; then uv venv --python python3 "$run027_dir/runtime/venv"; fi
run027_py="$run027_dir/runtime/venv/bin/python"
uv pip install --python "$run027_py" --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0
uv pip install --python "$run027_py" transformers==5.12.1 numpy==2.5.0 PyYAML==6.0.2 safetensors==0.8.0 ninja==1.11.1.4 setuptools==80.9.0 pip==25.2 pytest==9.0.2
"$run027_py" -m pip freeze > "$run027_dir/runtime/pip-freeze.txt"
nvcc --version > "$run027_dir/runtime/nvcc-version.txt"
nvidia-smi -q > "$run027_dir/runtime/nvidia-smi-q.txt"
lscpu > "$run027_dir/runtime/lscpu.txt"
"$run027_py" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(), torch.cuda.get_device_capability())'
