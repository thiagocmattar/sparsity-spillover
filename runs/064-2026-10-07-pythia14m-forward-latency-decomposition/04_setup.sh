#!/usr/bin/env bash
set -euo pipefail
RUN=/workspace/run064
mkdir -p "$RUN/runtime" /opt/run064-fast
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
export UV_CACHE_DIR=/workspace/run064-uv-cache UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5 UV_CONCURRENT_DOWNLOADS=2
python3 -m venv /opt/run064-fast/bootstrap
/opt/run064-fast/bootstrap/bin/pip install uv==0.8.15
export PATH=/opt/run064-fast/bootstrap/bin:$PATH
uv venv --python python3 /opt/run064-fast/venv
uv pip install --python /opt/run064-fast/venv/bin/python --index-strategy unsafe-best-match --index https://pypi.org/simple --index https://download.pytorch.org/whl/cu128 -r "$RUN/dependencies/run063/provenance/pip-freeze.txt"
export PATH=/opt/run064-fast/venv/bin:$PATH
uv pip freeze --python /opt/run064-fast/venv/bin/python > "$RUN/runtime/pip-freeze.txt"
nvcc --version > "$RUN/runtime/nvcc.txt"
nvidia-smi -q > "$RUN/runtime/nvidia-smi-initial.txt"
tar -xf /opt/run064-input.tar.gz -C /opt/run064-fast --wildcards 'dependencies/*'
export RUN064_BASE=/opt/run064-fast/dependencies/run063 RUN063_ARCHIVE=/opt/run064-fast/dependencies/run063/archive/root
export TORCH_EXTENSIONS_DIR=/opt/run064-fast/extensions TRITON_CACHE_DIR=/opt/run064-fast/triton MAX_JOBS=2 HF_HUB_OFFLINE=1
unset CUBLAS_WORKSPACE_CONFIG
python "$RUN064_BASE/01_prepare.py" verify
python -u "$RUN064_BASE/00_component.py"
cp "$RUN064_BASE/prelaunch/component-cuda.json" "$RUN/runtime/component-cuda.json"
python -u "$RUN064_BASE/00_precompile.py"
cp "$RUN064_BASE/runtime/precompile-status.json" "$RUN/runtime/precompile-status.json"
PYTHONPATH="$RUN:$RUN064_BASE" python -c 'import internal; internal.extension(False); internal.extension(True)'
touch "$RUN/runtime/setup-complete"
exec python -u "$RUN/03_execute.py"
