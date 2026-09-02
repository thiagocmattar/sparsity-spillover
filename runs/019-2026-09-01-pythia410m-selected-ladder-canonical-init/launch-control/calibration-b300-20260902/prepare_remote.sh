#!/usr/bin/env bash
set -euo pipefail

cd /workspace
printf '%s  %s\n' \
  '59eb0df48ad5bd42e8cde62bbb466c57ea30fa0ea38b00f0a47faa4fb715adf7' \
  'run019-b300-calibration.bundle' | sha256sum -c -
test ! -e sparsity-spillover
git clone run019-b300-calibration.bundle sparsity-spillover
git -C sparsity-spillover checkout --detach \
  7a0c49f54a5d0b89943228490908417952766905
test "$(git -C sparsity-spillover rev-parse HEAD)" = \
  '7a0c49f54a5d0b89943228490908417952766905'
mkdir -p \
  sparsity-spillover/data/tokenized/minipile-pythia-14m-full/train \
  sparsity-spillover/data/tokenized/minipile-pythia-14m-full/validation \
  sparsity-spillover/runs/019-2026-09-01-pythia410m-selected-ladder-canonical-init/prelaunch/initialization \
  run019-control
date -u +%FT%TZ > run019-control/input-transfer-started-utc.txt

