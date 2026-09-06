#!/usr/bin/env bash
set -euo pipefail
run025_base=/workspace/run025-autoresearch-rtxpro4500-001
run025_dir="$run025_base/sparsity-spillover/runs/025-2026-09-05-pythia-agentic-sparse-kernel-search"
run025_control="$run025_dir/artifacts/phase3-model-probes-rtxpro4500-001"
mkdir "$run025_control"
export PATH="$run025_dir/artifacts/runtime/venv-pythia/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export TORCH_EXTENSIONS_DIR="$run025_dir/artifacts/torch_extensions"
trap 'code=$?; printf "%s\n" "$code" > "$run025_control/exit-code.txt"; date -u +%FT%TZ > "$run025_control/finished-utc.txt"' EXIT
date -u +%FT%TZ > "$run025_control/started-utc.txt"
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
for run025_implementation in p0 k001 k003; do
  run_check "$run025_implementation-a0-all" python -u "$run025_dir/autoresearch/probe_models.py" \
    --condition 14m/a0 --implementation "$run025_implementation" --sites all \
    --attempt "$run025_implementation-14m-a0-all-rtxpro4500-001" --seconds 600
  run_check "$run025_implementation-a1h-active" python -u "$run025_dir/autoresearch/probe_models.py" \
    --condition 14m/a1h --implementation "$run025_implementation" --sites active \
    --attempt "$run025_implementation-14m-a1h-active-rtxpro4500-001" --seconds 600
done
run_check k002-a0 python -u "$run025_dir/autoresearch/probe_attention.py" \
  --condition 14m/a0 --attempt k002-14m-a0-rtxpro4500-001 --seconds 600 --sdpa-backend flash
run_check k002-a7-high python -u "$run025_dir/autoresearch/probe_attention.py" \
  --condition 14m/a7-0p5 --attempt k002-14m-a7-0p5-rtxpro4500-001 --seconds 600 --sdpa-backend flash
date -u +%FT%TZ > "$run025_control/all-checks-finished-utc.txt"
