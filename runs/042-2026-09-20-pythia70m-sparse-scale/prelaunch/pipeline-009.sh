#!/usr/bin/env bash
set -euo pipefail
cd /workspace/run042
while [ ! -f runtime/pipeline-008.exit ]; do
  test "$(date +%s)" -lt 1789924679
  sleep 10
done
test "$(cat runtime/pipeline-008.exit)" = 0
for candidate in opt015 opt016 opt017 opt018 opt020 opt021 opt030; do
  bash 05_execute.sh 27_qualify_regression.py "$candidate" >"runtime/regression-$candidate.log" 2>&1 || true
done
bash 05_execute.sh 03_execute.py --phase development --candidates opt015 opt016 opt017 opt018 opt020 opt021 opt030 \
  --tag 009 --deadline-epoch 1789926479 >runtime/development-009.log 2>&1
