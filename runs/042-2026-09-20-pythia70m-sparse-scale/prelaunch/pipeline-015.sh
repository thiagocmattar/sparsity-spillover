#!/usr/bin/env bash
set -euo pipefail
cd /workspace/run042
test "$(cat runtime/pipeline-014.exit)" = 0
for candidate in opt050 opt051 opt052; do
  bash 05_execute.sh candidates/check_short.py "$candidate" >"runtime/operator-$candidate.log" 2>&1 || true
  bash 05_execute.sh candidates/audit_short.py "$candidate" >"runtime/audit-$candidate.log" 2>&1
  bash 05_execute.sh 27_qualify_regression.py "$candidate" >"runtime/regression-$candidate.log" 2>&1 || true
done
bash 05_execute.sh 03_execute.py --phase development --candidates opt050 opt051 opt052 \
  --tag 015 --deadline-epoch 1789926479 >runtime/development-015.log 2>&1
