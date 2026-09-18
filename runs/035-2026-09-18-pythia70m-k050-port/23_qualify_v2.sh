#!/usr/bin/env bash
set -euo pipefail
cd /workspace/run035
while [ ! -f runtime/operators-004.exit ]; do sleep 5; done
test "$(cat runtime/operators-004.exit)" = 0
bash 05_execute.sh qualify 1789762688.225 002
