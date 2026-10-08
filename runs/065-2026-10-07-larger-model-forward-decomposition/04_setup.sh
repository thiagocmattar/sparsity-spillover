#!/usr/bin/env bash
set -euo pipefail
RUN=/workspace/run065
mkdir -p "$RUN/runtime" /opt/run065-fast
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
export UV_CACHE_DIR=/workspace/run065-uv-cache UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5 UV_CONCURRENT_DOWNLOADS=2
python3 -m venv /opt/run065-fast/bootstrap
/opt/run065-fast/bootstrap/bin/pip install uv==0.8.15
export PATH=/opt/run065-fast/bootstrap/bin:$PATH
uv venv --python python3 /opt/run065-fast/venv
uv pip install --python /opt/run065-fast/venv/bin/python --index-strategy unsafe-best-match --index https://pypi.org/simple --index https://download.pytorch.org/whl/cu128 -r "$RUN/provenance/pip-freeze.txt"
export PATH=/opt/run065-fast/venv/bin:$PATH
uv pip freeze --python /opt/run065-fast/venv/bin/python > "$RUN/runtime/pip-freeze.txt"
nvcc --version > "$RUN/runtime/nvcc.txt"
nvidia-smi -q > "$RUN/runtime/nvidia-smi-initial.txt"
tar -xf /opt/run065-input.tar.gz -C /opt/run065-fast --wildcards 'dependencies/*'
export RUN065_DEPS=/opt/run065-fast/dependencies/repository
export TORCH_EXTENSIONS_DIR=/opt/run065-fast/extensions TRITON_CACHE_DIR=/opt/run065-fast/triton MAX_JOBS=2 HF_HUB_OFFLINE=1 TORCH_CUDA_ARCH_LIST=12.0
unset CUBLAS_WORKSPACE_CONFIG
touch "$RUN/runtime/setup-complete"
exec python -u "$RUN/03_execute.py"
