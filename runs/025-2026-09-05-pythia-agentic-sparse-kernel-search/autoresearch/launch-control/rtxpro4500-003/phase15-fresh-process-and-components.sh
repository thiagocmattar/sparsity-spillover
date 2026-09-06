#!/usr/bin/env bash
set -euo pipefail

run025_base=/workspace/run025-autoresearch-rtxpro4500-001
run025_dir="$run025_base/sparsity-spillover/runs/025-2026-09-05-pythia-agentic-sparse-kernel-search"
run025_control="$run025_dir/artifacts/phase15-fresh-process-and-components-rtxpro4500-003"
mkdir "$run025_control"
export PATH="$run025_dir/artifacts/runtime/venv-pythia/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export TORCH_EXTENSIONS_DIR="$run025_dir/artifacts/torch_extensions"
trap 'code=$?; printf "%s\n" "$code" > "$run025_control/exit-code.txt"; date -u +%FT%TZ > "$run025_control/finished-utc.txt"' EXIT
date -u +%FT%TZ > "$run025_control/started-utc.txt"

run_probe() {
  local run025_condition="$1"
  local run025_implementation="$2"
  local run025_sites="$3"
  local run025_label="$4"
  local run025_program="$5"
  date -u +%FT%TZ > "$run025_control/$run025_label.started-utc.txt"
  set +e
  timeout --signal=TERM --kill-after=30s 600s python -u \
    "$run025_dir/autoresearch/$run025_program" \
    --condition "$run025_condition" --implementation "$run025_implementation" \
    --sites "$run025_sites" --execution eager --inputs 16 --passes 5 \
    --seconds 540 --full-validation --attempt "$run025_label" \
    > "$run025_control/$run025_label.log" 2>&1
  local run025_result=$?
  set -e
  printf '%s\n' "$run025_result" > "$run025_control/$run025_label.exit-code.txt"
  date -u +%FT%TZ > "$run025_control/$run025_label.finished-utc.txt"
}

matrix=(
  "14m/a0 k013 all 14m-a0 probe_k013_final.py"
  "14m/a1h k013 h 14m-a1h probe_k013_final.py"
  "14m/a4-0 k013 active 14m-a4-0 probe_k013_final.py"
  "14m/a4-0p01 k013 active 14m-a4-0p01 probe_k013_final.py"
  "14m/a4-0p05 k013 active 14m-a4-0p05 probe_k013_final.py"
  "14m/a4-0p1 k013 active 14m-a4-0p1 probe_k013_final.py"
  "14m/a4-0p5 k013 active 14m-a4-0p5 probe_k013_final.py"
  "14m/a7-0 k013 active 14m-a7-0 probe_k013_final.py"
  "14m/a7-0p01 k013 active 14m-a7-0p01 probe_k013_final.py"
  "14m/a7-0p05 k013 active 14m-a7-0p05 probe_k013_final.py"
  "14m/a7-0p1 k013 active 14m-a7-0p1 probe_k013_final.py"
  "14m/a7-0p5 k013 active 14m-a7-0p5 probe_k013_final.py"
  "70m/a0 k016 all 70m-a0 probe_k016_final.py"
  "70m/a1h k016 h 70m-a1h probe_k016_final.py"
  "70m/a4-0 k016 active 70m-a4-0 probe_k016_final.py"
  "70m/a4-0p01 k016 active 70m-a4-0p01 probe_k016_final.py"
  "70m/a4-0p05 k016 active 70m-a4-0p05 probe_k016_final.py"
  "70m/a4-0p1 k016 active 70m-a4-0p1 probe_k016_final.py"
  "70m/a4-0p5 k016 active 70m-a4-0p5 probe_k016_final.py"
  "70m/a7-0 k016 active 70m-a7-0 probe_k016_final.py"
  "70m/a7-0p01 k016 active 70m-a7-0p01 probe_k016_final.py"
  "70m/a7-0p05 k016 active 70m-a7-0p05 probe_k016_final.py"
  "70m/a7-0p1 k016 active 70m-a7-0p1 probe_k016_final.py"
  "70m/a7-0p5 k016 active 70m-a7-0p5 probe_k016_final.py"
  "410m/a0 k010 active 410m-a0 probe_frozen_final.py"
  "410m/a1h k010 active 410m-a1h probe_frozen_final.py"
  "410m/a4-0 k010 active 410m-a4-0 probe_frozen_final.py"
  "410m/a4-0p01 k010 active 410m-a4-0p01 probe_frozen_final.py"
  "410m/a4-0p05 k010 active 410m-a4-0p05 probe_frozen_final.py"
  "410m/a4-0p1 k010 active 410m-a4-0p1 probe_frozen_final.py"
  "410m/a4-0p5 k010 active 410m-a4-0p5 probe_frozen_final.py"
  "410m/a7-0 k010 active 410m-a7-0 probe_frozen_final.py"
  "410m/a7-0p01 k010 active 410m-a7-0p01 probe_frozen_final.py"
  "410m/a7-0p05 k010 active 410m-a7-0p05 probe_frozen_final.py"
  "410m/a7-0p1 k010 active 410m-a7-0p1 probe_frozen_final.py"
  "410m/a7-0p5 k010 active 410m-a7-0p5 probe_frozen_final.py"
)

for run025_repeat in 1 2 3; do
  for run025_spec in "${matrix[@]}"; do
    read -r run025_condition run025_implementation run025_sites run025_short run025_program <<< "$run025_spec"
    run_probe "$run025_condition" "$run025_implementation" "$run025_sites" \
      "fresh-r${run025_repeat}-${run025_short}-${run025_implementation}-rtxpro4500-003" "$run025_program"
  done
  date -u +%FT%TZ > "$run025_control/fresh-r${run025_repeat}-finished-utc.txt"
done

components=(
  "14m/a0 k013 14m-a0"
  "14m/a1h k013 14m-a1h"
  "14m/a4-0 k013 14m-a4-0"
  "14m/a4-0p5 k013 14m-a4-0p5"
  "14m/a7-0 k013 14m-a7-0"
  "14m/a7-0p5 k013 14m-a7-0p5"
  "70m/a1h k016 70m-a1h"
  "70m/a4-0 k016 70m-a4-0"
  "70m/a4-0p5 k016 70m-a4-0p5"
  "70m/a7-0 k016 70m-a7-0"
  "70m/a7-0p5 k016 70m-a7-0p5"
)

for run025_spec in "${components[@]}"; do
  read -r run025_condition run025_implementation run025_short <<< "$run025_spec"
  run_probe "$run025_condition" "$run025_implementation" active \
    "component-ffn-${run025_short}-${run025_implementation}-rtxpro4500-003" \
    probe_frozen_ffn.py
  if [[ "$run025_condition" != "14m/a1h" && "$run025_condition" != "70m/a1h" ]]; then
    run_probe "$run025_condition" "$run025_implementation" active \
      "component-attnproj-${run025_short}-${run025_implementation}-rtxpro4500-003" \
      probe_frozen_attention_projection.py
  fi
done

date -u +%FT%TZ > "$run025_control/all-checks-finished-utc.txt"
