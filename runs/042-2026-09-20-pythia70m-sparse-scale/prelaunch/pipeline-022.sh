#!/usr/bin/env bash
set -euo pipefail
cd /workspace/run042
while test ! -e runtime/pipeline-021.exit; do sleep 10; done
test "$(cat runtime/pipeline-021.exit)" = 0
bash 05_execute.sh candidates/check_attention.py opt073 >runtime/operator-opt073.log 2>&1
bash 05_execute.sh 03_execute.py --phase development --candidates opt073 --development-replicates 3 --tag 022 --deadline-epoch 1789926479 >runtime/development-022.log 2>&1
