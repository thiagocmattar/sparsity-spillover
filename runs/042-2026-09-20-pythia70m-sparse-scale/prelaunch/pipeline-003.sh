#!/usr/bin/env bash
set -euo pipefail
cd /workspace/run042
while [ ! -f runtime/pipeline-002.exit ]; do
  test "$(date +%s)" -lt 1789924679
  sleep 10
done
test "$(cat runtime/pipeline-002.exit)" = 0
bash 05_execute.sh candidates/check_joint.py opt019 >runtime/operator-opt019.log 2>&1 || true
for candidate in opt020 opt021; do
  bash 05_execute.sh candidates/check_short.py "$candidate" >"runtime/operator-$candidate.log" 2>&1 || true
done
bash 05_execute.sh 03_execute.py --phase development --candidates opt019 opt020 opt021 \
  --tag 004 --deadline-epoch 1789926479 >runtime/development-004.log 2>&1
