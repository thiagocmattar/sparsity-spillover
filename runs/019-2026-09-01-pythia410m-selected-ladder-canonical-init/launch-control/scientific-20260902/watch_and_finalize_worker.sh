#!/usr/bin/env bash
set -euo pipefail

condition=${1:?condition id required}
attempt_id=${2:?attempt id required}
archive=${3:?attempt archive path required}
teal_archive=${4:-}
repo=/workspace/sparsity-spillover
run_rel=runs/019-2026-09-01-pythia410m-selected-ladder-canonical-init
run_dir="$repo/$run_rel"
state_dir="/workspace/run019-control/$condition"

record_exit() {
    code=$?
    printf '%s\n' "$code" > "$state_dir/finalize-exit-code.txt"
    date -u +%FT%TZ > "$state_dir/finalize-finished-utc.txt"
}
trap record_exit EXIT

printf '%s\n' "$$" > "$state_dir/finalize-watcher.pid"
date -u +%FT%TZ > "$state_dir/finalize-watcher-started-utc.txt"
while test ! -f "$state_dir/train-exit-code.txt"; do
    sleep 60
done
test "$(cat "$state_dir/train-exit-code.txt")" = 0

case "$condition" in
    a0-gelu|a1h-relu)
        test -n "$teal_archive"
        date -u +%FT%TZ > "$state_dir/teal-started-utc.txt"
        teal_status=0
        cd "$repo"
        PYTHONPATH="$repo/src" /workspace/run019-venv/bin/python -u \
            "$run_dir/06_teal_posthoc.py" --condition "$condition" \
            > "$state_dir/teal.log" 2>&1 || teal_status=$?
        printf '%s\n' "$teal_status" > "$state_dir/teal-exit-code.txt"
        date -u +%FT%TZ > "$state_dir/teal-finished-utc.txt"
        test "$teal_status" = 0

        teal_dir="sparsity-spillover/$run_rel/artifacts/teal/$condition"
        teal_json="sparsity-spillover/$run_rel/artifacts/teal/$condition.json"
        cd /workspace
        find "$teal_dir" "$teal_json" -type f -print0 \
            | sort -z | xargs -0 sha256sum \
            > "$state_dir/teal-retrieval-files.sha256"
        ;;
    *)
        test -z "$teal_archive"
        ;;
esac

bash "$run_dir/launch-control/scientific-20260902/finalize_worker.sh" \
    "$condition" "$attempt_id" "$archive"

if test -n "$teal_archive"; then
    cd /workspace
    test ! -e "$teal_archive"
    tar -cf "$teal_archive" "$teal_dir" "$teal_json"
    sha256sum "$teal_archive" > "$state_dir/teal-archive.sha256"
fi
