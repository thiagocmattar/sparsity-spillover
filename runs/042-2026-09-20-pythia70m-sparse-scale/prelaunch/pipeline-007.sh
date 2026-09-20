#!/usr/bin/env bash
set -euo pipefail
cd /workspace/run042
while [ ! -f runtime/pipeline-006.exit ]; do
  test "$(date +%s)" -lt 1789924679
  sleep 10
done
test "$(cat runtime/pipeline-006.exit)" = 0
bash 05_execute.sh candidates/check_joint.py opt029 >runtime/operator-opt029.log 2>&1 || true
bash 05_execute.sh candidates/check_short.py opt030 >runtime/operator-opt030.log 2>&1 || true
for candidate in opt029 opt030; do
  bash 05_execute.sh candidates/audit_short.py "$candidate" >"runtime/stress-audit-$candidate.log" 2>&1
done
bash 05_execute.sh 03_execute.py --phase development --candidates opt029 opt030 \
  --tag 007 --deadline-epoch 1789926479 >runtime/development-007.log 2>&1
