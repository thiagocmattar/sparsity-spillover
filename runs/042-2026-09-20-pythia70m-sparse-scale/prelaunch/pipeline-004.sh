#!/usr/bin/env bash
set -euo pipefail
cd /workspace/run042
while [ ! -f runtime/pipeline-003.exit ]; do
  test "$(date +%s)" -lt 1789924679
  sleep 10
done
test "$(cat runtime/pipeline-003.exit)" = 0
for candidate in opt022 opt023 opt024; do
  bash 05_execute.sh candidates/check_attention.py "$candidate" >"runtime/operator-$candidate.log" 2>&1 || true
done
bash 05_execute.sh 03_execute.py --phase development --candidates opt022 opt023 opt024 \
  --tag 005 --deadline-epoch 1789926479 >runtime/development-005.log 2>&1
