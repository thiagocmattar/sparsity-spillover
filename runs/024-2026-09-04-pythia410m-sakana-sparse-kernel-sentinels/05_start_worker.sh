#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-/workspace/sparsity-spillover}"
STATE_ROOT="${2:-/workspace/run024-state}"
OUTPUT_ROOT="${3:-/workspace/run024-output}"
ATTEMPT_ID="${4:-001-sentinel-$(date -u +%Y%m%d-%H%M%S)}"
RUN_DIR="$REPO_ROOT/runs/024-2026-09-04-pythia410m-sakana-sparse-kernel-sentinels"
ATTEMPT_DIR="$OUTPUT_ROOT/attempts/$ATTEMPT_ID"
CONTROL_DIR="$OUTPUT_ROOT/control/$ATTEMPT_ID"
mkdir -p "$ATTEMPT_DIR" "$CONTROL_DIR"

if [[ "${RUN024_INNER:-0}" != "1" ]]; then
  if [[ -f "$CONTROL_DIR/worker.pid" ]] && kill -0 "$(cat "$CONTROL_DIR/worker.pid")" 2>/dev/null; then
    echo "A Run 024 worker is already active for $ATTEMPT_ID." >&2
    exit 1
  fi
  nohup env RUN024_INNER=1 bash "$0" "$REPO_ROOT" "$STATE_ROOT" "$OUTPUT_ROOT" "$ATTEMPT_ID" > "$CONTROL_DIR/worker.log" 2>&1 < /dev/null &
  WORKER_PID=$!
  printf '%s\n' "$WORKER_PID" > "$CONTROL_DIR/worker.pid"
  printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$CONTROL_DIR/started-utc.txt"
  echo "Started Run 024 attempt $ATTEMPT_ID as PID $WORKER_PID"
  echo "Log: $CONTROL_DIR/worker.log"
  exit 0
fi

CURRENT_STAGE="initializing"
finish() {
  code=$?
  printf '%s\n' "$code" > "$CONTROL_DIR/exit-code.txt"
  printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$CONTROL_DIR/finished-utc.txt"
  if [[ "$code" -eq 0 ]]; then printf 'complete\n' > "$CONTROL_DIR/status.txt"; else printf 'failed:%s\n' "$CURRENT_STAGE" > "$CONTROL_DIR/status.txt"; fi
  exit "$code"
}
trap finish EXIT
set_stage() {
  CURRENT_STAGE="$1"
  printf '%s\n' "$CURRENT_STAGE" > "$CONTROL_DIR/status.txt"
  printf '[stage] %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$CURRENT_STAGE"
}

set_stage setup
bash "$RUN_DIR/00_setup_remote.sh" "$REPO_ROOT" sentinel "$STATE_ROOT"
cp "$STATE_ROOT/static-preflight-patched.json" "$ATTEMPT_DIR/static-preflight.json"
cp "$STATE_ROOT/static-preflight-clean.json" "$ATTEMPT_DIR/static-preflight-official.json"
cp "$STATE_ROOT/applied-sakana-pythia410.patch" "$ATTEMPT_DIR/applied-sakana-pythia410.patch"
cp "$STATE_ROOT/applied-sakana-pythia410.patch.sha256" "$ATTEMPT_DIR/applied-sakana-pythia410.patch.sha256"
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

set_stage batch32_memory_calibration
"$STATE_ROOT/venv-pythia/bin/python" "$RUN_DIR/08_calibrate_batch.py" --upstream-dir "$DERIVED_DIR" --output "$ATTEMPT_DIR/batch-calibration.json"
INCLUDE_BATCH32="$("$STATE_ROOT/venv-pythia/bin/python" -c "import json; print('true' if json.load(open('$ATTEMPT_DIR/batch-calibration.json'))['eligible'] else 'false')")"
BATCH_PY="False"
if [[ "$INCLUDE_BATCH32" == "true" ]]; then BATCH_PY="True"; fi
"$STATE_ROOT/venv-pythia/bin/python" -c "import json; p='$ATTEMPT_DIR/batch-decision.json'; json.dump({'primary_batch':1,'include_batch32':$BATCH_PY,'calibration_file':'batch-calibration.json'},open(p,'w'),indent=2); open(p,'a').write('\\n')"
BATCH_FLAG=()
if [[ "$INCLUDE_BATCH32" == "true" ]]; then BATCH_FLAG=(--include-batch32); fi

set_stage pythia410m_sentinel_benchmark
"$STATE_ROOT/venv-pythia/bin/python" "$RUN_DIR/03_benchmark.py" --upstream-dir "$DERIVED_DIR" --phase sentinel --output-dir "$ATTEMPT_DIR" "${BATCH_FLAG[@]}"

set_stage verification
"$STATE_ROOT/venv-pythia/bin/python" "$RUN_DIR/04_verify.py" --attempt-dir "$ATTEMPT_DIR" --phase sentinel

set_stage packaging
(cd "$REPO_ROOT" && sha256sum \
  runs/024-2026-09-04-pythia410m-sakana-sparse-kernel-sentinels/*.py \
  runs/024-2026-09-04-pythia410m-sakana-sparse-kernel-sentinels/*.sh \
  runs/024-2026-09-04-pythia410m-sakana-sparse-kernel-sentinels/config.yaml \
  runs/023-2026-09-04-pythia70m-sakana-sparse-kernel-sentinels/{benchmark_core.py,pythia_sparse.py,02_remote_preflight.py,03_benchmark.py,sakana-pythia70.patch}) > "$ATTEMPT_DIR/benchmark-harness-sha256.txt"
cp "$CONTROL_DIR/worker.log" "$ATTEMPT_DIR/worker-through-verification.log"
RESULT_ARCHIVE="$OUTPUT_ROOT/run024-$ATTEMPT_ID-results.tar"
tar -cf "$RESULT_ARCHIVE" -C "$OUTPUT_ROOT" "attempts/$ATTEMPT_ID"
(cd "$OUTPUT_ROOT" && sha256sum "$(basename "$RESULT_ARCHIVE")" > "$(basename "$RESULT_ARCHIVE").sha256")
printf '%s\n' "$RESULT_ARCHIVE" > "$CONTROL_DIR/result-archive.txt"
set_stage complete
