#!/usr/bin/env bash
set -euo pipefail
run028_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Sequential development only; no benchmark overlaps another GPU workload.
for run028_candidate in k020 k021 k022; do
  for run028_condition in c25 c30; do
    run028_attempt="$run028_candidate-$run028_condition-dev-001"
    timeout --signal=TERM --kill-after=30s 600 bash "$run028_dir/07_trial.sh" \
      --attempt "$run028_attempt" --condition "$run028_condition" \
      --candidate "$run028_candidate" --inputs 8 --passes 3 \
      > "$run028_dir/runtime/$run028_attempt.log" 2>&1
  done
done
for run028_candidate in k020 k022; do
  timeout --signal=TERM --kill-after=30s 600 bash "$run028_dir/03_probe.sh" \
    --attempt "$run028_candidate-real-001" --candidate "$run028_candidate" \
    --conditions c25 c30 --inputs 2 > "$run028_dir/runtime/$run028_candidate-real-001.log" 2>&1
done
