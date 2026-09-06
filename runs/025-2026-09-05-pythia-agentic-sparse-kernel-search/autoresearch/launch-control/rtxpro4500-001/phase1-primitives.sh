#!/usr/bin/env bash
set -euo pipefail
run025_base=/workspace/run025-autoresearch-rtxpro4500-001
run025_dir="$run025_base/sparsity-spillover/runs/025-2026-09-05-pythia-agentic-sparse-kernel-search"
run025_control="$run025_dir/artifacts/phase1-primitives-rtxpro4500-001"
mkdir "$run025_control"
export PATH="$run025_dir/artifacts/runtime/venv-pythia/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export TORCH_EXTENSIONS_DIR="$run025_dir/artifacts/torch_extensions"
trap 'code=$?; printf "%s\n" "$code" > "$run025_control/exit-code.txt"; date -u +%FT%TZ > "$run025_control/finished-utc.txt"' EXIT
date -u +%FT%TZ > "$run025_control/started-utc.txt"
nvidia-smi -q > "$run025_control/nvidia-smi-before.txt"
run_check() {
  local run025_label="$1"
  shift
  date -u +%FT%TZ > "$run025_control/$run025_label.started-utc.txt"
  set +e
  timeout --signal=TERM --kill-after=30s 660s "$@" > "$run025_control/$run025_label.log" 2>&1
  local run025_result=$?
  set -e
  printf '%s\n' "$run025_result" > "$run025_control/$run025_label.exit-code.txt"
  date -u +%FT%TZ > "$run025_control/$run025_label.finished-utc.txt"
}
run_check k001-memcheck compute-sanitizer --tool memcheck --error-exitcode 99 \
  python -u "$run025_dir/autoresearch/candidates/k001/gpu_probe.py" \
  --output "$run025_dir/artifacts/k001-memcheck-rtxpro4500-001" --seconds 600
run_check k002-short python -u "$run025_dir/autoresearch/candidates/k002/qualify.py" \
  --attempt rtxpro4500-short-001 --max-length 129 --block-m 16
run_check k002-full python -u "$run025_dir/autoresearch/candidates/k002/qualify_contiguous.py" \
  --attempt rtxpro4500-contiguous-001 --seconds 600 --timing
run_check k003-memcheck compute-sanitizer --tool memcheck --error-exitcode 99 \
  python -u "$run025_dir/autoresearch/candidates/k003/gpu_probe.py" \
  --output "$run025_dir/artifacts/k003-memcheck-rtxpro4500-001" --seconds 600
nvidia-smi -q > "$run025_control/nvidia-smi-after.txt"
date -u +%FT%TZ > "$run025_control/all-checks-finished-utc.txt"
