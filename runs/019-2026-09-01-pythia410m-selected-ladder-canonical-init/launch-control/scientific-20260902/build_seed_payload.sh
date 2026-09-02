#!/usr/bin/env bash
set -euo pipefail

repo=/workspace/sparsity-spillover
control=/workspace/run019-control
archive=/workspace/run019-inputs.tar
run_rel=runs/019-2026-09-01-pythia410m-selected-ladder-canonical-init

test ! -e "$archive"
mkdir -p "$control"
cd "$repo"
tar -cf "$archive" \
  data/tokenized/minipile-pythia-14m-full/train/tokens.int32.bin \
  data/tokenized/minipile-pythia-14m-full/train/metadata.json \
  data/tokenized/minipile-pythia-14m-full/validation/tokens.int32.bin \
  data/tokenized/minipile-pythia-14m-full/validation/metadata.json \
  "$run_rel/prelaunch/initialization/pythia410m-seed1234.safetensors" \
  "$run_rel/prelaunch/initialization/pythia410m-seed1234-rng.pt" \
  "$run_rel/prelaunch/initialization/metadata.json"
sha256sum "$archive" > "$control/seed-payload.sha256"
wc -c "$archive" > "$control/seed-payload.bytes"
