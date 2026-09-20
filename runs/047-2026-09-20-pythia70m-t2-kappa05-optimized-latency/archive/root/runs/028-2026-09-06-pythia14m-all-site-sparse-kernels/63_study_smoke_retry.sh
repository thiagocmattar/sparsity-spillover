#!/usr/bin/env bash
set -euo pipefail
run028_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for run028_condition in c01 c30; do
  timeout --signal=TERM --kill-after=30s 300 bash "$run028_dir/61_study.sh" --condition "$run028_condition" --replicate 1 --attempt "study-$run028_condition-smoke-002" --smoke --policy preflight-policy-002.json > "$run028_dir/runtime/study-$run028_condition-smoke-002.log" 2>&1
done
