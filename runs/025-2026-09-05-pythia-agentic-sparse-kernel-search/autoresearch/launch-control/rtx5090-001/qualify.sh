#!/usr/bin/env bash
set -euo pipefail
run025_dir=/workspace/sparsity-spillover/runs/025-2026-09-05-pythia-agentic-sparse-kernel-search
run025_control="$run025_dir/artifacts/autoresearch-qualification-001"
mkdir "$run025_control"
export PATH="/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export TORCH_EXTENSIONS_DIR="$run025_dir/artifacts/torch_extensions"
trap 'code=$?; printf "%s\n" "$code" > "$run025_control/exit-code.txt"; date -u +%FT%TZ > "$run025_control/finished-utc.txt"' EXIT
date -u +%FT%TZ > "$run025_control/started-utc.txt"
timeout --signal=TERM --kill-after=30s 1500s bash "$run025_dir/00_setup_remote.sh" primitive
export PATH="$run025_dir/artifacts/runtime/venv-pythia/bin:$PATH"
run025_python="$run025_dir/artifacts/runtime/venv-pythia/bin/python"
run_check() {
  local label="$1"; shift
  date -u +%FT%TZ > "$run025_control/$label.started"
  set +e
  timeout --signal=TERM --kill-after=30s 660s "$@" > "$run025_control/$label.log" 2>&1
  local result=$?
  set -e
  printf '%s\n' "$result" > "$run025_control/$label.exit"
  date -u +%FT%TZ > "$run025_control/$label.finished"
}
run_check k001 compute-sanitizer --tool memcheck --error-exitcode 99 "$run025_python" -u "$run025_dir/autoresearch/candidates/k001/gpu_probe.py" --output "$run025_dir/artifacts/k001-memcheck-001" --seconds 600
run_check k002 "$run025_python" -u "$run025_dir/autoresearch/candidates/k002/qualify.py" --attempt cuda-short-001 --max-length 129 --block-m 16
run_check numerics "$run025_python" -u "$run025_dir/autoresearch/numerics/diagnose_layers.py" --condition 14m/a1h --blocks 1 --prefix-sweep --attempt a1h-layer-diagnostic-001 --seconds 600
run_check densegraph "$run025_python" -u "$run025_dir/autoresearch/dense_probe/probe.py" --condition 14m/a1h --attempt graph-001 --modes native graph --seconds 600
