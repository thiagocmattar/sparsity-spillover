#!/usr/bin/env bash
set -euo pipefail
RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
export UV_CACHE_DIR=/workspace/run061-uv-cache
mkdir -p "$RUN/runtime" /opt/run061-fast
python3 -m venv /opt/run061-fast/bootstrap
/opt/run061-fast/bootstrap/bin/pip install uv==0.8.15
export PATH=/opt/run061-fast/bootstrap/bin:$PATH
uv venv --python python3 /opt/run061-fast/venv
uv pip install --python /opt/run061-fast/venv/bin/python --index-strategy unsafe-best-match --index https://pypi.org/simple --index https://download.pytorch.org/whl/cu128 -r "$RUN/provenance/pip-freeze.txt"
export PATH=/opt/run061-fast/venv/bin:$PATH
uv pip freeze --python /opt/run061-fast/venv/bin/python > "$RUN/runtime/pip-freeze.txt"
nvcc --version > "$RUN/runtime/nvcc.txt"
nvidia-smi -q > "$RUN/runtime/nvidia-smi-initial.txt"
python "$RUN/01_prepare.py" verify
# Extract the same sealed vendor/source bytes onto local disk before first import.
tar -xf /workspace/run061-input.tar -C /opt/run061-fast --wildcards 'archive/*'
export RUN061_ARCHIVE=/opt/run061-fast/archive/root
python "$RUN/11_verify_cache.py"
export TORCH_EXTENSIONS_DIR=/opt/run061-fast/extensions TRITON_CACHE_DIR=/opt/run061-fast/triton
export MAX_JOBS=2 HF_HUB_OFFLINE=1
unset CUBLAS_WORKSPACE_CONFIG
python -u "$RUN/00_component.py"
cp "$RUN/prelaunch/component-cuda.json" "$RUN/runtime/component-cuda.json"
python -u "$RUN/00_precompile.py"
touch "$RUN/runtime/setup-complete"
