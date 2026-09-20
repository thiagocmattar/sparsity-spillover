#!/usr/bin/env bash
set -euo pipefail
cd /workspace/run042
test "$(cat runtime/pipeline-009.exit)" = 0
bash 05_execute.sh candidates/check_composition.py >runtime/operator-opt032.log 2>&1
bash 05_execute.sh 03_execute.py --phase development --candidates opt032 --development-replicates 3 \
  --tag 010 --deadline-epoch 1789926479 >runtime/development-010.log 2>&1
