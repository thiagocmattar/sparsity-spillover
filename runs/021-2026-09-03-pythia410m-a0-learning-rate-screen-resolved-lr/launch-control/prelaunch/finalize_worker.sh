#!/usr/bin/env bash
set -euo pipefail

condition=${1:?condition id required}
attempt_id=${2:?attempt id required}
archive=${3:?archive path required}
repo=/workspace/sparsity-spillover
run_rel=runs/021-2026-09-03-pythia410m-a0-learning-rate-screen-resolved-lr
run_dir="$repo/$run_rel"
attempt_rel="sparsity-spillover/$run_rel/artifacts/attempts/$attempt_id"
control=/workspace/run021-control

test "$(cat "$control/$condition/train-exit-code.txt")" = 0
test -f "$control/$condition/train-finished-utc.txt"
test -d "/workspace/$attempt_rel"
cd "$repo"
PYTHONPATH="$repo/src" /workspace/run021-venv/bin/python \
  "$run_dir/03_verify.py" --condition "$condition" \
  > "$control/$condition/verify.log" 2>&1
cd /workspace
find "run021-control" "$attempt_rel" -type f \
  ! -path "run021-control/$condition/retrieval-files.sha256" -print0 \
  | sort -z | xargs -0 sha256sum \
  > "$control/$condition/retrieval-files.sha256"
test ! -e "$archive"
tar -cf "$archive" run021-control "$attempt_rel"
sha256sum "$archive" > "$control/$condition/archive.sha256"
