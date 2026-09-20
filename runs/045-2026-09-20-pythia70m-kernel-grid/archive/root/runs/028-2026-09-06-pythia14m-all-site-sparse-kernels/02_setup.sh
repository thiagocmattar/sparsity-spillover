#!/usr/bin/env bash
set -euo pipefail
run028_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$run028_dir/runtime"
export UV_CACHE_DIR="$run028_dir/runtime/uv-cache"
export PATH="/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
command -v nvcc
command -v g++
python3 -c 'import torch; print(torch.__version__); print(torch.ones(1,device="cuda"))'
if ! command -v uv >/dev/null; then python3 -m pip install --break-system-packages uv==0.8.15; fi
# Deliberately isolated: the base image's Torch 2.8 is not the sealed 2.11 runtime.
if [[ ! -x "$run028_dir/runtime/venv/bin/python" ]]; then uv venv --python python3 "$run028_dir/runtime/venv"; fi
run028_py="$run028_dir/runtime/venv/bin/python"
uv pip install --python "$run028_py" --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0
uv pip install --python "$run028_py" transformers==5.12.1 numpy==2.5.0 PyYAML==6.0.2 safetensors==0.8.0 ninja==1.11.1.4 setuptools==80.9.0 pip==25.2 pytest==9.0.2
"$run028_py" -m pip freeze > "$run028_dir/runtime/pip-freeze.txt"
nvcc --version > "$run028_dir/runtime/nvcc-version.txt"
nvidia-smi -q > "$run028_dir/runtime/nvidia-smi-q.txt"
lscpu > "$run028_dir/runtime/lscpu.txt"
"$run028_py" -c 'import torch; print(torch.__version__,torch.cuda.get_device_name(),torch.ones(1,device="cuda"))'
