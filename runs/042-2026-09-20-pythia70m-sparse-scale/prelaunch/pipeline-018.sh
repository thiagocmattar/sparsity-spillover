#!/usr/bin/env bash
set -euo pipefail
cd /workspace/run042
test "$(cat runtime/pipeline-017.exit)" = 0
bash 05_execute.sh candidates/audit_parallel.py opt063 >runtime/audit-opt063.log 2>&1
bash 05_execute.sh 27_qualify_regression.py opt063 >runtime/regression-opt063.log 2>&1
bash 05_execute.sh 03_execute.py --phase development --candidates opt063 --development-replicates 3 \
  --tag 018 --deadline-epoch 1789926479 >runtime/development-018.log 2>&1
