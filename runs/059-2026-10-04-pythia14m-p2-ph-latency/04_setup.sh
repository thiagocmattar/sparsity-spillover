#!/usr/bin/env bash
set -euo pipefail
RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
export UV_CACHE_DIR=/workspace/run059-uv-cache
mkdir -p "$RUN/runtime"
python3 -m venv "$RUN/runtime/bootstrap"
"$RUN/runtime/bootstrap/bin/pip" install uv==0.8.15
export PATH="$RUN/runtime/bootstrap/bin:$PATH"
uv venv --python python3 "$RUN/runtime/venv"
uv pip install --python "$RUN/runtime/venv/bin/python" --index-strategy unsafe-best-match --index https://pypi.org/simple --index https://download.pytorch.org/whl/cu128 -r "$RUN/provenance/pip-freeze.txt"
uv pip freeze --python "$RUN/runtime/venv/bin/python" > "$RUN/runtime/pip-freeze.txt"
nvcc --version > "$RUN/runtime/nvcc.txt"
nvidia-smi -q > "$RUN/runtime/nvidia-smi-initial.txt"
"$RUN/runtime/venv/bin/python" "$RUN/01_prepare.py" verify
touch "$RUN/runtime/setup-complete"
