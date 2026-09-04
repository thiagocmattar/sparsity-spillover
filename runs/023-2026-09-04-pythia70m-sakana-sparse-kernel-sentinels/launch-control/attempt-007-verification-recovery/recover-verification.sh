#!/usr/bin/env bash
set -euo pipefail

RUN_DIR=/workspace/sparsity-spillover/runs/023-2026-09-04-pythia70m-sakana-sparse-kernel-sentinels
OUTPUT_ROOT=/workspace/run023-output
STATE_ROOT=/workspace/run023-state
SOURCE_ATTEMPT_ID=005-community-h100-nvl-20260904-1750
RECOVERY_ATTEMPT_ID=007-verification-recovery-20260904-1810
SOURCE_ATTEMPT_DIR="$OUTPUT_ROOT/attempts/$SOURCE_ATTEMPT_ID"
CONTROL_DIR="$OUTPUT_ROOT/control/$RECOVERY_ATTEMPT_ID"
VERIFY_DIR="$STATE_ROOT/$RECOVERY_ATTEMPT_ID"
PATCH_PATH="$RUN_DIR/launch-control/attempt-007-verification-recovery/verification-all-zero-rms.patch"
ARCHIVE="$OUTPUT_ROOT/run023-$SOURCE_ATTEMPT_ID.tar"

mkdir -p "$CONTROL_DIR" "$VERIFY_DIR"
date -u +%FT%TZ > "$CONTROL_DIR/started-utc.txt"
echo 'verification_recovery' > "$CONTROL_DIR/status.txt"

fail() {
  code=$?
  printf '%s\n' "$code" > "$CONTROL_DIR/exit-code.txt"
  date -u +%FT%TZ > "$CONTROL_DIR/finished-utc.txt"
  echo 'failed:verification_recovery' > "$CONTROL_DIR/status.txt"
  exit "$code"
}
trap fail ERR

cp "$RUN_DIR"/*.py "$RUN_DIR/config.yaml" "$VERIFY_DIR/"
patch --batch --forward -d "$VERIFY_DIR" -p1 < "$PATCH_PATH"
sha256sum "$RUN_DIR/04_verify.py" "$VERIFY_DIR/04_verify.py" "$PATCH_PATH" \
  > "$SOURCE_ATTEMPT_DIR/verification-recovery-attempt007-sha256.txt"
cp "$PATCH_PATH" "$SOURCE_ATTEMPT_DIR/verification-all-zero-rms.attempt007.patch"
cp "$VERIFY_DIR/04_verify.py" "$SOURCE_ATTEMPT_DIR/04_verify.attempt007.py"

PYTHONPATH="$VERIFY_DIR" "$STATE_ROOT/venv-pythia/bin/python" \
  "$VERIFY_DIR/04_verify.py" \
  --attempt-dir "$SOURCE_ATTEMPT_DIR" \
  --phase sentinel \
  > "$CONTROL_DIR/verification.log" 2>&1

tar -C "$OUTPUT_ROOT/attempts" -cf "$ARCHIVE" "$SOURCE_ATTEMPT_ID"
sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"
echo 'complete' > "$CONTROL_DIR/status.txt"
echo '0' > "$CONTROL_DIR/exit-code.txt"
date -u +%FT%TZ > "$CONTROL_DIR/finished-utc.txt"
trap - ERR
