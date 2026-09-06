#!/usr/bin/env bash
set -euo pipefail

run025_base=/workspace/run025-autoresearch-rtxpro4500-001
run025_dir="$run025_base/sparsity-spillover/runs/025-2026-09-05-pythia-agentic-sparse-kernel-search"
run025_control="$run025_dir/artifacts/phase4a-70m-a0-portfolio-rtxpro4500-002"
mkdir "$run025_control"
export PATH="$run025_dir/artifacts/runtime/venv-pythia/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export TORCH_EXTENSIONS_DIR="$run025_dir/artifacts/torch_extensions"
trap 'code=$?; printf "%s\n" "$code" > "$run025_control/exit-code.txt"; date -u +%FT%TZ > "$run025_control/finished-utc.txt"' EXIT
date -u +%FT%TZ > "$run025_control/started-utc.txt"

timeout --signal=TERM --kill-after=30s 660s python -u \
  "$run025_dir/autoresearch/probe_kernel_portfolio.py" \
  --condition 70m/a0 --sites all \
  --inputs 16 --passes 3 --seconds 600 \
  --attempt portfolio-70m-a0-all-rtxpro4500-002 \
  > "$run025_control/probe.log" 2>&1

date -u +%FT%TZ > "$run025_control/all-checks-finished-utc.txt"
