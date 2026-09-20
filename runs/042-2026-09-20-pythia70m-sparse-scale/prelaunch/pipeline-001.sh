#!/usr/bin/env bash
set -euo pipefail
cd /workspace/run042
while [ ! -f runtime/setup-001.exit ]; do
  test "$(date +%s)" -lt 1789925879
  sleep 5
done
test "$(cat runtime/setup-001.exit)" = 0
bash 05_execute.sh 10_test_operators.py >runtime/operators-001.log 2>&1
bash 05_execute.sh 06_control_checks.py >runtime/controls-001.log 2>&1
for condition in c00 c21 c16 m14-c01 m14-c20 m14-c35; do
  bash 05_execute.sh 02_benchmark.py --condition "$condition" --candidate full --replicate 1 --smoke \
    --attempt "smoke-$condition-full-r1-001" >"runtime/smoke-$condition.log" 2>&1
done
bash 05_execute.sh 03_execute.py --phase baseline --tag 001 --deadline-epoch 1789926479 >runtime/baseline-001.log 2>&1
for candidate in opt001 opt002 opt003 opt004 opt005 opt006; do
  bash 05_execute.sh candidates/check_joint.py "$candidate" >"runtime/operator-$candidate.log" 2>&1 || true
done
bash 05_execute.sh 03_execute.py --phase development --tag 001 --deadline-epoch 1789926479 >runtime/development-001.log 2>&1
for candidate in opt007 opt008 opt009 opt010 opt011; do
  bash 05_execute.sh candidates/check_attention.py "$candidate" >"runtime/operator-$candidate.log" 2>&1 || true
done
for candidate in opt012 opt013 opt014; do
  bash 05_execute.sh candidates/check_joint.py "$candidate" >"runtime/operator-$candidate.log" 2>&1 || true
done
bash 05_execute.sh 03_execute.py --phase development --candidates opt007 opt008 opt009 opt010 opt011 opt012 opt013 opt014 \
  --tag 002 --deadline-epoch 1789926479 >runtime/development-002.log 2>&1
