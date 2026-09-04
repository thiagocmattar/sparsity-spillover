#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-/workspace/sparsity-spillover}"
PHASE="${2:-sentinel}"
STATE_ROOT="${3:-/workspace/run023-state}"
OUTPUT_ROOT="${4:-/workspace/run023-output}"
ATTEMPT_ID="${5:-001-$PHASE-$(date -u +%Y%m%d-%H%M%S)}"
RUN_DIR="$REPO_ROOT/runs/023-2026-09-04-pythia70m-sakana-sparse-kernel-sentinels"
ATTEMPT_DIR="$OUTPUT_ROOT/attempts/$ATTEMPT_ID"
CONTROL_DIR="$OUTPUT_ROOT/control/$ATTEMPT_ID"

if [[ "$PHASE" != "sentinel" && "$PHASE" != "remainder" ]]; then
  echo "phase must be sentinel or remainder" >&2
  exit 2
fi
mkdir -p "$ATTEMPT_DIR" "$CONTROL_DIR"

if [[ "${RUN023_INNER:-0}" != "1" ]]; then
  if [[ -f "$CONTROL_DIR/worker.pid" ]] && kill -0 "$(cat "$CONTROL_DIR/worker.pid")" 2>/dev/null; then
    echo "A Run 023 worker is already active for $ATTEMPT_ID." >&2
    exit 1
  fi
  nohup env RUN023_INNER=1 bash "$0" "$REPO_ROOT" "$PHASE" "$STATE_ROOT" "$OUTPUT_ROOT" "$ATTEMPT_ID" > "$CONTROL_DIR/worker.log" 2>&1 < /dev/null &
  WORKER_PID=$!
  printf '%s\n' "$WORKER_PID" > "$CONTROL_DIR/worker.pid"
  printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$CONTROL_DIR/started-utc.txt"
  echo "Started Run 023 $PHASE attempt $ATTEMPT_ID as PID $WORKER_PID"
  echo "Log: $CONTROL_DIR/worker.log"
  exit 0
fi

CURRENT_STAGE="initializing"
finish() {
  code=$?
  printf '%s\n' "$code" > "$CONTROL_DIR/exit-code.txt"
  printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$CONTROL_DIR/finished-utc.txt"
  if [[ "$code" -eq 0 ]]; then
    printf 'complete\n' > "$CONTROL_DIR/status.txt"
  else
    printf 'failed:%s\n' "$CURRENT_STAGE" > "$CONTROL_DIR/status.txt"
  fi
  exit "$code"
}
trap finish EXIT

set_stage() {
  CURRENT_STAGE="$1"
  printf '%s\n' "$CURRENT_STAGE" > "$CONTROL_DIR/status.txt"
  printf '[stage] %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$CURRENT_STAGE"
}

set_stage setup
bash "$RUN_DIR/00_setup_remote.sh" "$REPO_ROOT" "$PHASE" "$STATE_ROOT"
cp "$STATE_ROOT/static-preflight-patched.json" "$ATTEMPT_DIR/static-preflight.json"
cp "$STATE_ROOT/static-preflight-clean.json" "$ATTEMPT_DIR/static-preflight-official.json"
cp "$STATE_ROOT/applied-sakana-pythia70.patch" "$ATTEMPT_DIR/applied-sakana-pythia70.patch"
cp "$STATE_ROOT/applied-sakana-pythia70.patch.sha256" "$ATTEMPT_DIR/applied-sakana-pythia70.patch.sha256"
cp "$STATE_ROOT/pythia-pip-freeze.txt" "$ATTEMPT_DIR/pythia-pip-freeze.txt"
cp "$STATE_ROOT/upstream-pip-freeze.txt" "$ATTEMPT_DIR/upstream-pip-freeze.txt"
cp "$STATE_ROOT/nvidia-smi-q.txt" "$ATTEMPT_DIR/nvidia-smi-q.txt"
cp "$STATE_ROOT/upstream-model-revision.txt" "$ATTEMPT_DIR/upstream-model-revision.txt"
cp "$STATE_ROOT/upstream-model-sha256.txt" "$ATTEMPT_DIR/upstream-model-sha256.txt"
OFFICIAL_DIR="$(cat "$STATE_ROOT/upstream-official-dir.txt")"
DERIVED_DIR="$(cat "$STATE_ROOT/upstream-derived-dir.txt")"
MODEL_DIR="$(cat "$STATE_ROOT/upstream-model-dir.txt")"

set_stage official_upstream_positive_control
(cd "$OFFICIAL_DIR" && "$STATE_ROOT/venv-upstream/bin/python" benchmark_inference.py --model-path "$MODEL_DIR" --out-csv "$ATTEMPT_DIR/upstream-positive-control.csv" --batch-size 64 --seq-len 2048 --dtype bf16 --device cuda --reps 50 --warmup-reps 5)

set_stage derived_kernel_preflight
"$STATE_ROOT/venv-pythia/bin/python" "$RUN_DIR/02_remote_preflight.py" --upstream-dir "$DERIVED_DIR" --output "$ATTEMPT_DIR/remote-preflight.json"

set_stage pythia70m_phase_benchmark
"$STATE_ROOT/venv-pythia/bin/python" "$RUN_DIR/03_benchmark.py" --upstream-dir "$DERIVED_DIR" --phase "$PHASE" --output-dir "$ATTEMPT_DIR"

set_stage verification
"$STATE_ROOT/venv-pythia/bin/python" "$RUN_DIR/04_verify.py" --attempt-dir "$ATTEMPT_DIR" --phase "$PHASE"

set_stage packaging
cp "$CONTROL_DIR/worker.log" "$ATTEMPT_DIR/worker-through-verification.log"
RESULT_ARCHIVE="$OUTPUT_ROOT/run023-$ATTEMPT_ID-results.tar"
tar -cf "$RESULT_ARCHIVE" -C "$OUTPUT_ROOT" "attempts/$ATTEMPT_ID"
(cd "$OUTPUT_ROOT" && sha256sum "$(basename "$RESULT_ARCHIVE")" > "$(basename "$RESULT_ARCHIVE").sha256")
printf '%s\n' "$RESULT_ARCHIVE" > "$CONTROL_DIR/result-archive.txt"
set_stage complete
