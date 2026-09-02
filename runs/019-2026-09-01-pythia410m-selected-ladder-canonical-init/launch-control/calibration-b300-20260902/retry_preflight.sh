#!/usr/bin/env bash
set -uo pipefail

repo=/workspace/sparsity-spillover
control=/workspace/run019-control
run_dir="$repo/runs/019-2026-09-01-pythia410m-selected-ladder-canonical-init"
python=/workspace/run019-venv/bin/python
commit=71f9c4709c4ca091215b9e1baf219557666145ed

slug=retry-preflight-attempt04
status=0
date -u +%FT%TZ > "$control/$slug-started-utc.txt"
{
  if ! test "$(git -C "$repo" rev-parse HEAD)" = "$commit"; then
    status=10
  fi
  if (( status == 0 )); then
    "$python" -m pip install --no-cache-dir pytest==9.1.1 || status=$?
  fi
  if (( status == 0 )); then
    cd "$repo" || status=$?
  fi
  if (( status == 0 )); then
    mkdir -p "$repo/.pytest_tmp" || status=$?
  fi
  if (( status == 0 )); then
    "$python" -m pytest -p no:cacheprovider "$run_dir/test_run019.py" --tb=short || status=$?
  fi
  if (( status == 0 )); then
    "$python" "$run_dir/08_verify_initialization.py" \
      --output "$control/initialization-verification-attempt04.json" || status=$?
  fi
} > "$control/$slug.log" 2>&1
printf '%s\n' "$status" > "$control/$slug-exit-code.txt"
date -u +%FT%TZ > "$control/$slug-finished-utc.txt"
exit "$status"
