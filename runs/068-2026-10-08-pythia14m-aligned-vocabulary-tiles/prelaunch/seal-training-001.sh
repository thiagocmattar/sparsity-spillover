#!/usr/bin/env bash
set -eu
control=/workspace/run068-control
run=/workspace/sparsity-spillover/runs/068-2026-10-08-pythia14m-aligned-vocabulary-tiles
test "$(cat "$control/pipeline.exit")" = 0
test -f "$run/artifacts/verification.json"
test ! -e "$control/seal-training.pid"
nohup setsid bash -c '
  /workspace/run068-venv/bin/python /workspace/sparsity-spillover/runs/068-2026-10-08-pythia14m-aligned-vocabulary-tiles/15_seal_training.py
  result=$?
  printf "%s\n" "$result" > /workspace/run068-control/seal-training.exit
  exit "$result"
' > "$control/seal-training.log" 2>&1 < /dev/null &
echo $! > "$control/seal-training.pid"
cat "$control/seal-training.pid"
