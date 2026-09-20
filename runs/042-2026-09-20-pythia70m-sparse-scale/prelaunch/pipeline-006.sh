#!/usr/bin/env bash
set -euo pipefail
cd /workspace/run042
while [ ! -f runtime/pipeline-005.exit ]; do
  test "$(date +%s)" -lt 1789924679
  sleep 10
done
test "$(cat runtime/pipeline-005.exit)" = 0
for candidate in opt001 opt015 opt016 opt017 opt018 opt020 opt021; do
  bash 05_execute.sh candidates/audit_short.py "$candidate" >"runtime/stress-audit-$candidate.log" 2>&1
done
