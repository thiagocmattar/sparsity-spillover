#!/usr/bin/env bash
set -euo pipefail
cd /workspace/run040
deadline=1789917919.103
while [ ! -e runtime/setup-001.exit ]; do sleep 10; done
test "$(cat runtime/setup-001.exit)" = 0
date -u +%FT%TZ > runtime/operators-started
bash 05_execute.sh operators > runtime/operators-001.log 2>&1
date -u +%FT%TZ > runtime/smoke-started
bash 05_execute.sh smoke "$deadline" 001 > runtime/smoke-001.log 2>&1
date -u +%FT%TZ > runtime/scientific-started
bash 05_execute.sh scientific "$deadline" 001 > runtime/scientific-001.log 2>&1
date -u +%FT%TZ > runtime/diagnostic-complete
