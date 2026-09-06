#!/usr/bin/env bash
set -euo pipefail

run025_base=/workspace/run025-autoresearch-rtxpro4500-001
run025_dir="$run025_base/sparsity-spillover/runs/025-2026-09-05-pythia-agentic-sparse-kernel-search"
run025_control="$run025_dir/artifacts/phase6b-14m-k011-mask-search-rtxpro4500-002"
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
  local run025_label="k011search-14m-$run025_short-$run025_mask-rtxpro4500-002"
  date -u +%FT%TZ > "$run025_control/$run025_label.started-utc.txt"
  set +e
  timeout --signal=TERM --kill-after=30s 660s env RUN025_K011_MASK="$run025_mask" \
    python -u "$run025_dir/autoresearch/probe_k011_models.py" \
    --condition "$run025_condition" --implementation k011 --sites active \
    --execution eager --inputs 16 --passes 3 --seconds 600 \
    --attempt "$run025_label" > "$run025_control/$run025_label.log" 2>&1
  local run025_result=$?
  set -e
  printf '%s\n' "$run025_result" > "$run025_control/$run025_label.exit-code.txt"
  date -u +%FT%TZ > "$run025_control/$run025_label.finished-utc.txt"
}

for run025_mask in prefix1 prefix2 prefix3 prefix4 prefix5 suffix1 suffix2 suffix3 suffix4 suffix5 even odd; do
  run_probe 14m/a7-0 a7-0 "$run025_mask"
  run_probe 14m/a4-0p5 a4-0p5 "$run025_mask"
  run_probe 14m/a7-0p5 a7-0p5 "$run025_mask"
done

date -u +%FT%TZ > "$run025_control/all-checks-finished-utc.txt"
