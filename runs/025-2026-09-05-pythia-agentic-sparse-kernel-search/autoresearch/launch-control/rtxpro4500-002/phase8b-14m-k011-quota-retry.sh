#!/usr/bin/env bash
set -euo pipefail

run025_base=/workspace/run025-autoresearch-rtxpro4500-001
run025_dir="$run025_base/sparsity-spillover/runs/025-2026-09-05-pythia-agentic-sparse-kernel-search"
run025_prior="$run025_dir/artifacts/phase8a-14m-k011-complete-validation-mask-search-rtxpro4500-002"
run025_control="$run025_dir/artifacts/phase8b-14m-k011-quota-retry-rtxpro4500-002"
mkdir "$run025_control"
export PATH="$run025_dir/artifacts/runtime/venv-pythia/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export TORCH_EXTENSIONS_DIR="$run025_dir/artifacts/torch_extensions"
trap 'code=$?; printf "%s\n" "$code" > "$run025_control/exit-code.txt"; date -u +%FT%TZ > "$run025_control/finished-utc.txt"' EXIT
date -u +%FT%TZ > "$run025_control/started-utc.txt"

require_complete() {
  python - "$1" <<'PY'
import json
from pathlib import Path
import sys

directory = Path(sys.argv[1])
status = json.loads((directory / "status.json").read_text())
validation = json.loads((directory / "full-validation.json").read_text())
if status.get("stage") != "complete":
    raise SystemExit("prior artifact is not complete")
if validation.get("blocks") != 338 or validation.get("excluded_tail_tokens") != 1444:
    raise SystemExit("prior artifact has wrong validation coverage")
if not all(validation["pass"].values()):
    raise SystemExit("prior artifact is not qualified")
PY
}

run_probe() {
  local run025_condition="$1"
  local run025_sites="$2"
  local run025_short="$3"
  local run025_mask="$4"
  local run025_label="k011retry-14m-$run025_short-$run025_mask-rtxpro4500-002"
  date -u +%FT%TZ > "$run025_control/$run025_label.started-utc.txt"
  set +e
  timeout --signal=TERM --kill-after=30s 720s python -u \
    "$run025_dir/autoresearch/probe_k011_full_validation.py" \
    --condition "$run025_condition" --mask "$run025_mask" --sites "$run025_sites" \
    --inputs 16 --passes 3 --seconds 660 --full-validation \
    --attempt "$run025_label" > "$run025_control/$run025_label.log" 2>&1
  local run025_result=$?
  set -e
  printf '%s\n' "$run025_result" > "$run025_control/$run025_label.exit-code.txt"
  date -u +%FT%TZ > "$run025_control/$run025_label.finished-utc.txt"
  return "$run025_result"
}

# Suffix3 A0 and A4-kappa-0 completed before the quota interruption. Verify
# rather than repeat them; A7-kappa-0 is a new infrastructure attempt.
require_complete "$run025_dir/artifacts/k011full-14m-a0-suffix3-rtxpro4500-002"
require_complete "$run025_dir/artifacts/k011full-14m-a4-0-suffix3-rtxpro4500-002"
if run_probe 14m/a7-0 active a7-0 suffix3; then
  printf 'screen-passed\n' > "$run025_control/suffix3.gate.txt"
  run025_failed=0
  run_probe 14m/a1h h a1h suffix3 || run025_failed=1
  run_probe 14m/a4-0p5 active a4-0p5 suffix3 || run025_failed=1
  run_probe 14m/a7-0p5 active a7-0p5 suffix3 || run025_failed=1
  if [ "$run025_failed" -eq 0 ]; then
    printf 'all-six-passed\n' > "$run025_control/suffix3.gate.txt"
  else
    printf 'confirmation-failed\n' > "$run025_control/suffix3.gate.txt"
  fi
else
  printf 'screen-failed-confirmation-skipped\n' > "$run025_control/suffix3.gate.txt"
fi

for run025_mask in suffix4 odd; do
  run025_screen_failed=0
  run_probe 14m/a0 all a0 "$run025_mask" || run025_screen_failed=1
  run_probe 14m/a4-0 active a4-0 "$run025_mask" || run025_screen_failed=1
  run_probe 14m/a7-0 active a7-0 "$run025_mask" || run025_screen_failed=1
  if [ "$run025_screen_failed" -eq 0 ]; then
    printf 'screen-passed\n' > "$run025_control/$run025_mask.gate.txt"
    run025_confirmation_failed=0
    run_probe 14m/a1h h a1h "$run025_mask" || run025_confirmation_failed=1
    run_probe 14m/a4-0p5 active a4-0p5 "$run025_mask" || run025_confirmation_failed=1
    run_probe 14m/a7-0p5 active a7-0p5 "$run025_mask" || run025_confirmation_failed=1
    if [ "$run025_confirmation_failed" -eq 0 ]; then
      printf 'all-six-passed\n' > "$run025_control/$run025_mask.gate.txt"
    else
      printf 'confirmation-failed\n' > "$run025_control/$run025_mask.gate.txt"
    fi
  else
    printf 'screen-failed-confirmation-skipped\n' > "$run025_control/$run025_mask.gate.txt"
  fi
done

date -u +%FT%TZ > "$run025_control/all-checks-finished-utc.txt"
