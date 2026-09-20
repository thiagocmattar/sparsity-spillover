#!/usr/bin/env bash
set -euo pipefail
cd /workspace/run042
while test ! -e runtime/pipeline-020.exit; do sleep 10; done
test "$(cat runtime/pipeline-020.exit)" = 0
for candidate in opt070 opt071 opt072; do
 bash 05_execute.sh candidates/audit_parallel.py "$candidate" >"runtime/audit-$candidate.log" 2>&1 || true
 bash 05_execute.sh 27_qualify_regression.py "$candidate" >"runtime/regression-$candidate.log" 2>&1 || true
done
bash 05_execute.sh 03_execute.py --phase development --candidates opt070 opt071 opt072 --tag 021 --deadline-epoch 1789926479 >runtime/development-021.log 2>&1
