#!/usr/bin/env bash
set -uo pipefail

expected_commit=${1:?expected source commit required}
bundle_sha256=${2:?source bundle sha256 required}
payload_sha256=${3:?input payload sha256 required}
repo=/workspace/sparsity-spillover
control=/workspace/run020-control
bundle=/workspace/run020-science.bundle
payload=/workspace/run020-inputs.tar
run_rel=runs/020-2026-09-03-pythia410m-a0-learning-rate-screen

status=0
mkdir -p "$control"
test ! -e "$control/setup-started-utc.txt" || exit 90
date -u +%FT%TZ > "$control/setup-started-utc.txt"
{
  if ! echo "$bundle_sha256  $bundle" | sha256sum -c -; then status=11; fi
  if (( status == 0 )) && ! echo "$payload_sha256  $payload" | sha256sum -c -; then status=12; fi
  if (( status == 0 )) && test -e "$repo"; then status=13; fi
  if (( status == 0 )); then git clone "$bundle" "$repo" || status=$?; fi
  if (( status == 0 )); then git -C "$repo" checkout --detach "$expected_commit" || status=$?; fi
  if (( status == 0 )) && ! test "$(git -C "$repo" rev-parse HEAD)" = "$expected_commit"; then status=14; fi
  if (( status == 0 )); then tar -xf "$payload" -C "$repo" || status=$?; fi
  if (( status == 0 )); then bash "$repo/$run_rel/01_setup_remote.sh" || status=$?; fi
} > "$control/setup.log" 2>&1
printf '%s\n' "$status" > "$control/setup-exit-code.txt"
date -u +%FT%TZ > "$control/setup-finished-utc.txt"
exit "$status"
