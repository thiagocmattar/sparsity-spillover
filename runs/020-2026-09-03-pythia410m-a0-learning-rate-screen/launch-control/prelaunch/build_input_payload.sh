#!/usr/bin/env bash
set -euo pipefail

repo=${1:-/workspace/sparsity-spillover}
archive=${2:-/workspace/run020-inputs.tar}
control=${3:-/workspace/run020-control}
run019=runs/019-2026-09-01-pythia410m-selected-ladder-canonical-init

test ! -e "$archive"
mkdir -p "$control"
cd "$repo"
tar -cf "$archive" \
  data/tokenized/minipile-pythia-14m-full/train/tokens.int32.bin \
  data/tokenized/minipile-pythia-14m-full/train/metadata.json \
  data/tokenized/minipile-pythia-14m-full/validation/tokens.int32.bin \
  data/tokenized/minipile-pythia-14m-full/validation/metadata.json \
  "$run019/prelaunch/initialization/pythia410m-seed1234.safetensors" \
  "$run019/prelaunch/initialization/pythia410m-seed1234-rng.pt" \
  "$run019/prelaunch/initialization/metadata.json"
sha256sum "$archive" > "$control/input-payload.sha256"
wc -c "$archive" > "$control/input-payload.bytes"
