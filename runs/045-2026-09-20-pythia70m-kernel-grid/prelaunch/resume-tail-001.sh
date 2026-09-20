#!/usr/bin/env bash
# Infrastructure handoff only; the original deadline and scientific files stay fixed.
set -uo pipefail
cd /workspace/run045
while test ! -f /workspace/run045-control/pipeline.exit; do
  if test "$(date +%s)" -ge 1789939835; then exit 124; fi
  sleep 30
done
if test "$(cat /workspace/run045-control/pipeline.exit)" = 0; then
  echo 'Original pipeline completed; no continuation needed.'
  exit 0
fi
export RUN045_DEADLINE_EPOCH=1789941035
bash 05_execute.sh prelaunch/resume_tail.py --deadline-epoch 1789941035
run045_result=$?
printf '%s\n' "$run045_result" > /workspace/run045-control/tail.exit
exit "$run045_result"
