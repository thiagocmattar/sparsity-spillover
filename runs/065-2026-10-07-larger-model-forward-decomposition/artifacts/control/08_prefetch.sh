#!/usr/bin/env bash
set -euo pipefail
# Prefetch the identical frozen environment while checkpoint ingress proceeds.
# This environment is separate from the authoritative pipeline environment.
mkdir -p /opt/run065-prefetch /workspace/run065-control
export UV_CACHE_DIR=/workspace/run065-uv-cache UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5 UV_CONCURRENT_DOWNLOADS=2
python3 -m venv /opt/run065-prefetch/bootstrap
/opt/run065-prefetch/bootstrap/bin/pip install uv==0.8.15
export PATH=/opt/run065-prefetch/bootstrap/bin:$PATH
uv venv --python python3 /opt/run065-prefetch/venv
uv pip install --python /opt/run065-prefetch/venv/bin/python --index-strategy unsafe-best-match --index https://pypi.org/simple --index https://download.pytorch.org/whl/cu128 -r /workspace/run065-control/pip-freeze.txt
uv pip freeze --python /opt/run065-prefetch/venv/bin/python > /workspace/run065-control/prefetch-freeze.txt
touch /workspace/run065-control/prefetch-complete
