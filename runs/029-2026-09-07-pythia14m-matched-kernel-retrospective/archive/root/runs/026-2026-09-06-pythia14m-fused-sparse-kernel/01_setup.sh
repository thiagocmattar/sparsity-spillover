#!/usr/bin/env bash
set -euo pipefail
run026_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$run026_dir/runtime"
export UV_CACHE_DIR="$run026_dir/runtime/uv-cache"
export PATH="/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
command -v nvcc
command -v g++
python3 -c 'import sys; assert sys.version_info[:2] == (3,12)'
if ! command -v uv >/dev/null; then python3 -m pip install --break-system-packages uv==0.8.15; fi
if [[ ! -x "$run026_dir/runtime/venv/bin/python" ]]; then uv venv --python python3 "$run026_dir/runtime/venv"; fi
run026_py="$run026_dir/runtime/venv/bin/python"
uv pip install --python "$run026_py" --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0
uv pip install --python "$run026_py" transformers==5.12.1 numpy==2.5.0 PyYAML==6.0.2 safetensors==0.8.0 ninja==1.11.1.4 setuptools==80.9.0 pip==25.2 pytest==9.0.2
"$run026_py" -m pip freeze > "$run026_dir/runtime/pip-freeze.txt"
nvcc --version > "$run026_dir/runtime/nvcc-version.txt"
"$run026_py" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(), torch.cuda.get_device_capability())'
