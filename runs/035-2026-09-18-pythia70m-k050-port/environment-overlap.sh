#!/usr/bin/env bash
set -euo pipefail
RUN035="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
export UV_CACHE_DIR=/workspace/run035-uv-cache UV_HTTP_TIMEOUT=120
mkdir -p "$RUN035/runtime"
python3 -m venv "$RUN035/runtime/bootstrap"
"$RUN035/runtime/bootstrap/bin/pip" install uv==0.8.15
export PATH="$RUN035/runtime/bootstrap/bin:$PATH"
uv venv --python python3 "$RUN035/runtime/venv"
uv pip install --python "$RUN035/runtime/venv/bin/python" --index-url https://pypi.org/simple --find-links https://download.pytorch.org/whl/cu128/torch/ -r "$RUN035/provenance/pip-freeze.txt"
uv pip freeze --python "$RUN035/runtime/venv/bin/python" > "$RUN035/runtime/pip-freeze.txt"
nvcc --version > "$RUN035/runtime/nvcc.txt"
nvidia-smi -q > "$RUN035/runtime/nvidia-smi-initial.txt"
while [ ! -f "$RUN035/runtime/inputs-ready" ]; do sleep 5; done
"$RUN035/runtime/venv/bin/python" "$RUN035/01_prepare.py" verify
touch "$RUN035/runtime/setup-complete"
