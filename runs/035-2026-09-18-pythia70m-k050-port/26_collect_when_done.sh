#!/usr/bin/env bash
set -euo pipefail
cd /workspace/run035
while [ ! -f runtime/scientific.exit ]; do sleep 30; done
test "$(cat runtime/scientific.exit)" = 0
runtime/venv/bin/python -u 08_collect.py
