#!/usr/bin/env bash
set -euo pipefail
cd /workspace/run042
while [ ! -f runtime/pipeline-004.exit ]; do
  test "$(date +%s)" -lt 1789924679
  sleep 10
done
test "$(cat runtime/pipeline-004.exit)" = 0
for candidate in opt025 opt026 opt027 opt028; do
  bash 05_execute.sh candidates/check_head.py "$candidate" >"runtime/operator-$candidate.log" 2>&1 || true
done
bash 05_execute.sh 03_execute.py --phase development --candidates opt025 opt026 opt027 opt028 \
  --tag 006 --deadline-epoch 1789926479 >runtime/development-006.log 2>&1
