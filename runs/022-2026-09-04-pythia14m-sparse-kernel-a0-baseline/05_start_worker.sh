#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-/workspace/sparsity-spillover}"
STATE_ROOT="${2:-/workspace/run022-state}"
OUTPUT_ROOT="${3:-/workspace/run022-output}"
ATTEMPT_ID="${4:-001-$(date -u +%Y%m%d-%H%M%S)}"
RUN_DIR="$REPO_ROOT/runs/022-2026-09-04-pythia14m-sparse-kernel-a0-baseline"
ATTEMPT_DIR="$OUTPUT_ROOT/attempts/$ATTEMPT_ID"
CONTROL_DIR="$OUTPUT_ROOT/control/$ATTEMPT_ID"

mkdir -p "$ATTEMPT_DIR" "$CONTROL_DIR"

if [[ "${RUN022_INNER:-0}" != "1" ]]; then
  if [[ -f "$CONTROL_DIR/worker.pid" ]] && kill -0 "$(cat "$CONTROL_DIR/worker.pid")" 2>/dev/null; then
    echo "A Run 022 worker is already active for $ATTEMPT_ID." >&2
    exit 1
  fi
  nohup env RUN022_INNER=1 bash "$0" "$REPO_ROOT" "$STATE_ROOT" "$OUTPUT_ROOT" "$ATTEMPT_ID" \
    > "$CONTROL_DIR/worker.log" 2>&1 < /dev/null &
  WORKER_PID=$!
  printf '%s\n' "$WORKER_PID" > "$CONTROL_DIR/worker.pid"
  printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$CONTROL_DIR/started-utc.txt"
  echo "Started Run 022 attempt $ATTEMPT_ID as PID $WORKER_PID"
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
bash "$RUN_DIR/00_setup_remote.sh" "$REPO_ROOT" "$STATE_ROOT"
cp "$STATE_ROOT/static-preflight.json" "$ATTEMPT_DIR/static-preflight.json"
cp "$STATE_ROOT/pythia-pip-freeze.txt" "$ATTEMPT_DIR/pythia-pip-freeze.txt"
cp "$STATE_ROOT/upstream-pip-freeze.txt" "$ATTEMPT_DIR/upstream-pip-freeze.txt"
cp "$STATE_ROOT/nvidia-smi-q.txt" "$ATTEMPT_DIR/nvidia-smi-q.txt"
cp "$STATE_ROOT/upstream-model-revision.txt" "$ATTEMPT_DIR/upstream-model-revision.txt"
cp "$STATE_ROOT/upstream-model-sha256.txt" "$ATTEMPT_DIR/upstream-model-sha256.txt"
UPSTREAM_DIR="$(cat "$STATE_ROOT/upstream-dir.txt")"
MODEL_DIR="$(cat "$STATE_ROOT/upstream-model-dir.txt")"

set_stage upstream_positive_control
(
  cd "$UPSTREAM_DIR"
  "$STATE_ROOT/venv-upstream/bin/python" benchmark_inference.py \
    --model-path "$MODEL_DIR" \
    --out-csv "$ATTEMPT_DIR/upstream-positive-control.csv" \
    --batch-size 64 --seq-len 2048 --dtype bf16 --device cuda \
    --reps 50 --warmup-reps 5
)

set_stage raw_ell_preflight
"$STATE_ROOT/venv-pythia/bin/python" "$RUN_DIR/02_remote_preflight.py" \
  --upstream-dir "$UPSTREAM_DIR" --output "$ATTEMPT_DIR/remote-preflight.json"

set_stage pythia_a0_benchmark
"$STATE_ROOT/venv-pythia/bin/python" "$RUN_DIR/03_benchmark.py" \
  --upstream-dir "$UPSTREAM_DIR" --output-dir "$ATTEMPT_DIR"

set_stage verification
"$STATE_ROOT/venv-pythia/bin/python" "$RUN_DIR/04_verify.py" --attempt-dir "$ATTEMPT_DIR"

set_stage packaging
cp "$CONTROL_DIR/worker.log" "$ATTEMPT_DIR/worker-through-verification.log"
RESULT_ARCHIVE="$OUTPUT_ROOT/run022-$ATTEMPT_ID-results.tar"
tar -cf "$RESULT_ARCHIVE" -C "$OUTPUT_ROOT" "attempts/$ATTEMPT_ID"
(cd "$OUTPUT_ROOT" && sha256sum "$(basename "$RESULT_ARCHIVE")" > "$(basename "$RESULT_ARCHIVE").sha256")
printf '%s\n' "$RESULT_ARCHIVE" > "$CONTROL_DIR/result-archive.txt"
set_stage complete
