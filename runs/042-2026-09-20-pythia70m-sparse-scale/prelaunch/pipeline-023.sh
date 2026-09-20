#!/usr/bin/env bash
set -euo pipefail
cd /workspace/run042
while test ! -e runtime/pipeline-022.exit; do sleep 10; done
test "$(cat runtime/pipeline-022.exit)" = 0
bash 05_execute.sh candidates/audit_parallel.py opt074 >runtime/audit-opt074.log 2>&1
bash 05_execute.sh 27_qualify_regression.py opt074 >runtime/regression-opt074.log 2>&1
bash 05_execute.sh 03_execute.py --phase development --candidates opt074 --development-replicates 3 --tag 023 --deadline-epoch 1789926479 >runtime/development-023.log 2>&1
