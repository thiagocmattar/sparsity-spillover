#!/usr/bin/env bash
set -euo pipefail
RUN041="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export UV_CACHE_DIR=/workspace/run041-latency-uv-cache
mkdir -p "$RUN041/runtime"
python3 -m venv "$RUN041/runtime/bootstrap"
"$RUN041/runtime/bootstrap/bin/pip" install uv==0.8.15
export PATH="$RUN041/runtime/bootstrap/bin:$PATH"
uv venv --python python3 "$RUN041/runtime/venv"
uv pip install --python "$RUN041/runtime/venv/bin/python" --index-url https://pypi.org/simple --find-links https://download.pytorch.org/whl/cu128/torch/ -r "$RUN041/provenance/pip-freeze.txt"
uv pip freeze --python "$RUN041/runtime/venv/bin/python" > "$RUN041/runtime/pip-freeze.txt"
nvcc --version > "$RUN041/runtime/nvcc.txt"
nvidia-smi -q > "$RUN041/runtime/nvidia-smi-initial.txt"
"$RUN041/runtime/venv/bin/python" "$RUN041/01_prepare.py" verify
touch "$RUN041/runtime/setup-complete"
