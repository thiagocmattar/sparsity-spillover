#!/usr/bin/env bash
set -euo pipefail
cd /workspace/run042
while [ ! -f runtime/pipeline-001.exit ]; do
  test "$(date +%s)" -lt 1789924679
  sleep 10
done
test "$(cat runtime/pipeline-001.exit)" = 0
for candidate in opt015 opt016 opt017 opt018; do
  bash 05_execute.sh candidates/check_short.py "$candidate" >"runtime/operator-$candidate.log" 2>&1 || true
done
bash 05_execute.sh 03_execute.py --phase development --candidates opt015 opt016 opt017 opt018 \
  --tag 003 --deadline-epoch 1789926479 >runtime/development-003.log 2>&1
