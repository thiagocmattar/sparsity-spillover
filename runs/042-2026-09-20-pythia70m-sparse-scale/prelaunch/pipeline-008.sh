#!/usr/bin/env bash
set -euo pipefail
cd /workspace/run042
while [ ! -f runtime/pipeline-007.exit ]; do
  test "$(date +%s)" -lt 1789924679
  sleep 10
done
test "$(cat runtime/pipeline-007.exit)" = 0
bash 05_execute.sh candidates/check_attention.py opt031 >runtime/operator-opt031.log 2>&1 || true
bash 05_execute.sh 03_execute.py --phase development --candidates opt031 \
  --tag 008 --deadline-epoch 1789926479 >runtime/development-008.log 2>&1
