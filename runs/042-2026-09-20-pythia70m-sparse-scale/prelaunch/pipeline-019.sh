#!/usr/bin/env bash
set -euo pipefail
cd /workspace/run042
test "$(cat runtime/pipeline-018.exit)" = 0
bash 05_execute.sh candidates/check_attention.py opt064 >runtime/operator-opt064.log 2>&1
bash 05_execute.sh 03_execute.py --phase development --candidates opt064 --development-replicates 3 --tag 019 --deadline-epoch 1789926479 >runtime/development-019.log 2>&1
