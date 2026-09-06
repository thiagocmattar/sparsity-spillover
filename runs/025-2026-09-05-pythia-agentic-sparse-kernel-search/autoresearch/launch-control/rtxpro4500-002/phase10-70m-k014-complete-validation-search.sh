#!/usr/bin/env bash
set -euo pipefail

run025_base=/workspace/run025-autoresearch-rtxpro4500-001
run025_dir="$run025_base/sparsity-spillover/runs/025-2026-09-05-pythia-agentic-sparse-kernel-search"
run025_control="$run025_dir/artifacts/phase10-70m-k014-complete-validation-search-rtxpro4500-002"
mkdir "$run025_control"
export PATH="$run025_dir/artifacts/runtime/venv-pythia/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export TORCH_EXTENSIONS_DIR="$run025_dir/artifacts/torch_extensions"
trap 'code=$?; printf "%s\n" "$code" > "$run025_control/exit-code.txt"; date -u +%FT%TZ > "$run025_control/finished-utc.txt"' EXIT
date -u +%FT%TZ > "$run025_control/started-utc.txt"

run_probe() {
  local run025_condition="$1"
  local run025_short="$2"
  local run025_mask="$3"
  local run025_attempt="k014full-70m-$run025_short-$run025_mask-rtxpro4500-002"
  date -u +%FT%TZ > "$run025_control/$run025_attempt.started-utc.txt"
  set +e
  timeout --signal=TERM --kill-after=30s 900s python -u \
    "$run025_dir/autoresearch/probe_k014_models.py" \
    --condition "$run025_condition" --implementation "k014-$run025_mask" \
    --sites active --execution eager --inputs 16 --passes 3 --seconds 840 \
    --full-validation --attempt "$run025_attempt" \
    > "$run025_control/$run025_attempt.log" 2>&1
  local run025_result=$?
  set -e
  printf '%s\n' "$run025_result" > "$run025_control/$run025_attempt.exit-code.txt"
  date -u +%FT%TZ > "$run025_control/$run025_attempt.finished-utc.txt"
}

for run025_mask in suffix2 suffix3 suffix4 suffix5 all6 odd; do
  run_probe 70m/a1h a1h "$run025_mask"
  run_probe 70m/a4-0 a4-0 "$run025_mask"
  run_probe 70m/a4-0p5 a4-0p5 "$run025_mask"
  run_probe 70m/a7-0 a7-0 "$run025_mask"
  run_probe 70m/a7-0p5 a7-0p5 "$run025_mask"
done

run025_selection="$run025_dir/artifacts/k014-complete-validation-selection-rtxpro4500-002/selection.json"
python -u "$run025_dir/autoresearch/select_k014_complete_validation.py" \
  --artifacts "$run025_dir/artifacts" --output "$run025_selection" \
  > "$run025_control/selection.stdout.json"
date -u +%FT%TZ > "$run025_control/all-checks-finished-utc.txt"
