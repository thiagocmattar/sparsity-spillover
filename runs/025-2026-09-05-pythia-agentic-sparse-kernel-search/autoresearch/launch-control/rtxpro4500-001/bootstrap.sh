#!/usr/bin/env bash
set -euo pipefail
run025_base=/workspace/run025-autoresearch-rtxpro4500-001
run025_repo="$run025_base/sparsity-spillover"
run025_dir="$run025_repo/runs/025-2026-09-05-pythia-agentic-sparse-kernel-search"
run025_control="$run025_dir/artifacts/bootstrap-rtxpro4500-001"
mkdir -p "$run025_control"
export PATH="/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export TORCH_EXTENSIONS_DIR="$run025_dir/artifacts/torch_extensions"
trap 'code=$?; printf "%s\n" "$code" > "$run025_control/exit-code.txt"; date -u +%FT%TZ > "$run025_control/finished-utc.txt"' EXIT
date -u +%FT%TZ > "$run025_control/started-utc.txt"
timeout --signal=TERM --kill-after=30s 1500s bash "$run025_dir/00_setup_remote.sh" primitive
date -u +%FT%TZ > "$run025_control/ready-utc.txt"
