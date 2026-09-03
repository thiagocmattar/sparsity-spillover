#!/usr/bin/env bash
set -uo pipefail

condition=${1:?condition id required}
repo=/workspace/sparsity-spillover
run_dir="$repo/runs/021-2026-09-03-pythia410m-a0-learning-rate-screen-resolved-lr"
control="/workspace/run021-control/$condition"

test "$(cat /workspace/run021-control/preflight-$condition-exit-code.txt)" = 0 || exit 91
test ! -e "$control" || exit 92
mkdir -p "$control"
date -u +%FT%TZ > "$control/train-started-utc.txt"
status=0
cd "$repo" || exit 98
PYTHONPATH="$repo/src" /workspace/run021-venv/bin/python -u \
  "$run_dir/02_train.py" --worker "$condition" \
  > "$control/train.log" 2>&1 || status=$?
printf '%s\n' "$status" > "$control/train-exit-code.txt"
date -u +%FT%TZ > "$control/train-finished-utc.txt"
exit "$status"
