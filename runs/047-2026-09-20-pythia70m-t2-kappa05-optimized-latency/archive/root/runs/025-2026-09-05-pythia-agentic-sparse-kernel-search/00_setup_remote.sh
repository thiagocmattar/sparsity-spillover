#!/usr/bin/env bash
# Official CUDA 12.8.1 PyTorch template; intentionally separate pinned Pythia env.
set -euo pipefail
run025_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
run025_phase="${1:-calibration}"
[[ "$run025_phase" == calibration || "$run025_phase" == primitive ]] || exit 2
run025_state="$run025_dir/artifacts/runtime"
mkdir -p -- "$run025_state"
export UV_CACHE_DIR="$run025_state/uv-cache"
export PIP_CACHE_DIR="$run025_state/pip-cache"
export HF_HOME="$run025_state/hf-cache"
python3 -c 'import sys; assert sys.version_info[:2] == (3,12), sys.version'
command -v nvcc
command -v g++
command -v timeout
command -v setsid
# The template torch 2.8 is deliberately NOT the scientific runtime. This is
# the justified exception to inheriting a template's preinstalled torch.
if ! command -v uv >/dev/null; then
  python3 -m pip install --break-system-packages uv==0.8.15
fi
if [[ ! -x "$run025_state/venv-pythia/bin/python" ]]; then
  uv venv --python python3 "$run025_state/venv-pythia"
fi
run025_python="$run025_state/venv-pythia/bin/python"
uv pip install --python "$run025_python" --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0
uv pip install --python "$run025_python" transformers==5.12.1 numpy==2.5.0 PyYAML==6.0.2 safetensors==0.8.0 ninja==1.11.1.4 setuptools==80.9.0 pip==25.2
"$run025_python" "$run025_dir/02_package.py" --verify "$run025_dir/prelaunch/transfer-$run025_phase.json"
"$run025_python" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(), torch.cuda.get_device_capability())'
"$run025_python" -m pip freeze > "$run025_state/pip-freeze.txt"
nvcc --version > "$run025_state/nvcc-version.txt"
printf 'Ready: %s\n' "$run025_python"
