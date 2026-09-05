#!/usr/bin/env bash
set -euo pipefail

REPO=/workspace/sparsity-spillover
RUN_NAME=024-2026-09-04-pythia410m-sakana-sparse-kernel-sentinels
RUN_DIR="$REPO/runs/$RUN_NAME"
ATTEMPT_ID=007-community-h100-nvl-canonical-eager-20260905-0039
ATTEMPT="/workspace/run024-output/attempts/$ATTEMPT_ID"
ORIGINAL_CONTROL="/workspace/run024-output/control/$ATTEMPT_ID"
RECOVERY_CONTROL=/workspace/run024-output/control/008-community-h100-nvl-cross-device-attention-coverage-recovery
SOURCE_CONTROL="$RUN_DIR/launch-control/attempt-008-community-h100-nvl-cross-device-attention-coverage"
UPSTREAM=/workspace/run024-state/sparser-faster-llms-derived
VENV=/workspace/run024-state/venv-pythia
ARCHIVE="/workspace/run024-output/run024-$ATTEMPT_ID-recovered-results.tar.gz"
PATCH="$ATTEMPT/cross-device-attention-coverage.patch"

mkdir -p "$RECOVERY_CONTROL" "$SOURCE_CONTROL"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RECOVERY_CONTROL/started-utc.txt"
echo running > "$RECOVERY_CONTROL/status.txt"

finish() {
    local exit_code=$?
    date -u +%Y-%m-%dT%H:%M:%SZ > "$RECOVERY_CONTROL/finished-utc.txt"
    printf '%s\n' "$exit_code" > "$RECOVERY_CONTROL/exit-code.txt"
    if [[ $exit_code -ne 0 ]]; then
        echo failed > "$RECOVERY_CONTROL/status.txt"
    fi
}
trap finish EXIT

test ! -e "$ATTEMPT/a7-ol1-kappa-0p5.json"
cp "$ATTEMPT/recover_final_condition.py" "$SOURCE_CONTROL/recover_final_condition.py"
cp "$ATTEMPT/cross-device-attention-coverage.patch" "$SOURCE_CONTROL/cross-device-attention-coverage.patch"
cp "$ATTEMPT/smoke_cross_device_coverage.py" "$SOURCE_CONTROL/smoke_cross_device_coverage.py"

cd "$ATTEMPT"
sha256sum \
    a0-gelu.json \
    a1h-relu.json \
    a4-ol1-kappa-0.json \
    a4-ol1-kappa-0p5.json \
    a7-ol1-kappa-0.json \
    > pre-recovery-existing-results-sha256.txt

cd "$REPO"
if git apply --check "$PATCH"; then
    git apply "$PATCH"
elif git apply --reverse --check "$PATCH"; then
    echo "coverage patch already applied"
else
    echo "coverage patch is neither cleanly applicable nor already applied" >&2
    exit 2
fi
git apply --reverse --check "$PATCH"

source "$VENV/bin/activate"
export PATH="$VENV/bin:$PATH"
export PYTHONUNBUFFERED=1
python -m py_compile \
    "$RUN_DIR/03_benchmark.py" \
    "$RUN_DIR/04_verify.py" \
    "$SOURCE_CONTROL/recover_final_condition.py" \
    "$SOURCE_CONTROL/smoke_cross_device_coverage.py"
python "$SOURCE_CONTROL/smoke_cross_device_coverage.py"

python "$SOURCE_CONTROL/recover_final_condition.py" \
    --repo-root "$REPO" \
    --upstream-dir "$UPSTREAM" \
    --output-dir "$ATTEMPT" \
    --original-control-dir "$ORIGINAL_CONTROL" \
    --preexisting-manifest "$ATTEMPT/pre-recovery-existing-results-sha256.txt"

python "$RUN_DIR/04_verify.py" --attempt-dir "$ATTEMPT" --phase sentinel

cd "$ATTEMPT"
sha256sum \
    a0-gelu.json \
    a1h-relu.json \
    a4-ol1-kappa-0.json \
    a4-ol1-kappa-0p5.json \
    a7-ol1-kappa-0.json \
    > post-recovery-existing-results-sha256.txt
cmp pre-recovery-existing-results-sha256.txt post-recovery-existing-results-sha256.txt
sha256sum \
    a0-gelu.json \
    a1h-relu.json \
    a4-ol1-kappa-0.json \
    a4-ol1-kappa-0p5.json \
    a7-ol1-kappa-0.json \
    a7-ol1-kappa-0p5.json \
    cohort.json \
    verification.json \
    > verified-results-sha256.txt
sha256sum \
    "$RUN_DIR/03_benchmark.py" \
    "$RUN_DIR/04_verify.py" \
    "$SOURCE_CONTROL/recover_final_condition.py" \
    "$SOURCE_CONTROL/smoke_cross_device_coverage.py" \
    "$PATCH" \
    > final-harness-sha256.txt
cp "$ORIGINAL_CONTROL/started-utc.txt" original-attempt-started-utc.txt
cp "$ORIGINAL_CONTROL/finished-utc.txt" original-attempt-finished-utc.txt
cp "$ORIGINAL_CONTROL/exit-code.txt" original-attempt-exit-code.txt
cp "$ORIGINAL_CONTROL/status.txt" original-attempt-status.txt
date -u +%Y-%m-%dT%H:%M:%SZ > recovery-science-finished-utc.txt
echo verified > "$RECOVERY_CONTROL/status.txt"
cp "$RECOVERY_CONTROL/started-utc.txt" recovery-started-utc.txt
cp "$RECOVERY_CONTROL/status.txt" recovery-status-at-packaging.txt
cp "$RECOVERY_CONTROL/worker.log" recovery-worker.log

cd /workspace/run024-output/attempts
tar -czf "$ARCHIVE" "$ATTEMPT_ID"
sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"
echo complete > "$RECOVERY_CONTROL/status.txt"
echo "PASS: recovery, verification, and archive complete: $ARCHIVE"
