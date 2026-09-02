#!/usr/bin/env bash
set -uo pipefail

condition=${1:?condition id required}
expected_commit=${2:?expected source commit required}
control=/workspace/run019-control
script=/workspace/sparsity-spillover/runs/019-2026-09-01-pythia410m-selected-ladder-canonical-init/launch-control/scientific-20260902/verify_worker.sh
status=0

test ! -e "$control/preflight-$condition-started-utc.txt" || exit 90
date -u +%FT%TZ > "$control/preflight-$condition-started-utc.txt"
bash "$script" "$condition" "$expected_commit" \
  > "$control/preflight-$condition.log" 2>&1 || status=$?
printf '%s\n' "$status" > "$control/preflight-$condition-exit-code.txt"
date -u +%FT%TZ > "$control/preflight-$condition-finished-utc.txt"
exit "$status"
