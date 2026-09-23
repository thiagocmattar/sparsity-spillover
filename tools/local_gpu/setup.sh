#!/usr/bin/env bash
# Infrastructure setup only; no experiment or benchmark is launched.
set -euo pipefail
test "${WSL_DISTRO_NAME:-}" = SparsityGPU || { echo 'Run inside the dedicated SparsityGPU WSL distribution.' >&2; exit 1; }
test "$(id -u)" = 0 || { echo 'Run setup as root.' >&2; exit 1; }
source /etc/os-release
test "$ID:$VERSION_ID" = ubuntu:24.04
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SOURCE_DIR/../.." && pwd)"
TASK_PREFIX=/opt/sparsity-gpu
mkdir -p "$TASK_PREFIX/logs"
exec > >(tee -a "$TASK_PREFIX/logs/setup.log") 2>&1
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends ca-certificates curl build-essential python3.12 python3.12-venv
curl --fail --location --retry 3 \
  https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
  -o "$TASK_PREFIX/cuda-keyring.deb"
dpkg -i "$TASK_PREFIX/cuda-keyring.deb"
apt-get update
# Compiler, headers and toolkit only; never install a Linux GPU driver in WSL.
apt-get install -y --no-install-recommends cuda-toolkit-12-8
python3.12 -m venv "$TASK_PREFIX/bootstrap"
"$TASK_PREFIX/bootstrap/bin/pip" install uv==0.8.15
"$TASK_PREFIX/bootstrap/bin/uv" venv --python python3.12 --allow-existing "$TASK_PREFIX/venv"
"$TASK_PREFIX/bootstrap/bin/uv" pip install --python "$TASK_PREFIX/venv/bin/python" \
  --index-url https://pypi.org/simple --find-links https://download.pytorch.org/whl/cu128/torch/ \
  -r "$REPO_DIR/runs/049-2026-09-22-pythia70m-short-row-limits/provenance/pip-freeze.txt"
cat > "$TASK_PREFIX/activate.sh" <<'EOF'
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/opt/sparsity-gpu/venv/bin:$CUDA_HOME/bin:$PATH
export TORCH_EXTENSIONS_DIR=/opt/sparsity-gpu/extensions
export TRITON_CACHE_DIR=/opt/sparsity-gpu/triton-cache
export MAX_JOBS=4
EOF
source "$TASK_PREFIX/activate.sh"
python -m pip freeze > "$TASK_PREFIX/logs/pip-freeze.txt"
nvcc --version > "$TASK_PREFIX/logs/nvcc.txt"
nvidia-smi -q > "$TASK_PREFIX/logs/nvidia-smi.txt"
python "$SOURCE_DIR/smoke.py" --output "$TASK_PREFIX/logs/smoke.json"
printf 'Local GPU environment and infrastructure smoke are ready.\n'
