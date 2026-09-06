#!/usr/bin/env bash
set -euo pipefail

run025_base=/workspace/run025-autoresearch-rtxpro4500-001
run025_repo="$run025_base/sparsity-spillover"
run025_run="$run025_repo/runs/025-2026-09-05-pythia-agentic-sparse-kernel-search"
run025_incoming=/workspace/run025-phase17-rtxpro4500-004
run025_bootstrap="$run025_incoming/bootstrap-control"
mkdir -p "$run025_bootstrap"
trap 'code=$?; printf "%s\n" "$code" > "$run025_bootstrap/exit-code.txt"; date -u +%FT%TZ > "$run025_bootstrap/finished-utc.txt"' EXIT
date -u +%FT%TZ > "$run025_bootstrap/started-utc.txt"

cd "$run025_incoming"
sha256sum -c SHA256SUMS > "$run025_bootstrap/incoming-sha256.txt"
test -d "$run025_repo"
test -x "$run025_run/artifacts/runtime/venv-pythia/bin/python"
tar -xzf run025-phase17-code.tar.gz -C "$run025_repo"
date -u +%FT%TZ > "$run025_bootstrap/code-overlaid-utc.txt"

export PATH="$run025_run/artifacts/runtime/venv-pythia/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export TORCH_EXTENSIONS_DIR="$run025_run/artifacts/torch_extensions"
run025_python="$run025_run/artifacts/runtime/venv-pythia/bin/python"
"$run025_python" -m py_compile \
  "$run025_run/autoresearch/probe_models.py" \
  "$run025_run/autoresearch/probe_k004_flagged_z_models.py" \
  "$run025_run/autoresearch/probe_k009_models.py" \
  "$run025_run/autoresearch/probe_k013_final.py" \
  "$run025_run/autoresearch/probe_k016_final.py" \
  "$run025_run/autoresearch/probe_frozen_final.py"
"$run025_python" - <<'PY'
import torch

assert torch.cuda.is_available()
major, minor = torch.cuda.get_device_capability()
assert major == 12, (major, minor)
print(torch.__version__, torch.cuda.get_device_name(), (major, minor))
PY
test ! -e "$run025_run/artifacts/phase17-fixed-rmodel-pairs-rtxpro4500-004"
"$run025_python" -m pip freeze > "$run025_bootstrap/pip-freeze.txt"
nvidia-smi -q > "$run025_bootstrap/nvidia-smi-q.txt"
nvcc --version > "$run025_bootstrap/nvcc-version.txt"
date -u +%FT%TZ > "$run025_bootstrap/runtime-ready-utc.txt"

timeout --signal=TERM --kill-after=30s 6300s bash \
  "$run025_run/autoresearch/launch-control/rtxpro4500-004/phase17-fixed-rmodel-pairs.sh"

run025_phase="$run025_run/artifacts/phase17-fixed-rmodel-pairs-rtxpro4500-004"
mkdir "$run025_phase/runtime-provenance"
cp "$run025_bootstrap/pip-freeze.txt" "$run025_phase/runtime-provenance/"
cp "$run025_bootstrap/nvidia-smi-q.txt" "$run025_phase/runtime-provenance/"
cp "$run025_bootstrap/nvcc-version.txt" "$run025_phase/runtime-provenance/"
cp "$run025_bootstrap/incoming-sha256.txt" "$run025_phase/runtime-provenance/"
date -u +%FT%TZ > "$run025_phase/runtime-provenance/packaged-utc.txt"
date -u +%FT%TZ > "$run025_bootstrap/evidence-ready-utc.txt"
