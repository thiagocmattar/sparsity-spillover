#!/usr/bin/env bash
# One independent condition per GPU. Preflight state is discarded before training.
set -euo pipefail
repo=/workspace/sparsity-spillover
run="$repo/runs/058-2026-10-03-pythia14m-t2-p2-versus-ph"
py=/workspace/run058-venv/bin/python
cd "$repo"
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
mkdir -p "$run/prelaunch" /workspace/run058-control
mapfile -t workers < <("$py" -c "import sys;sys.path.insert(0,'$run');from run_config import load_config;print('\n'.join(load_config()['runpod']['worker_assignments']))")
if [[ ${#workers[@]} != 4 ]]; then echo 'Expected four workers' >&2; exit 1; fi
phase() {
    local script="$1" label="$2" failed=0
    local -a pids=()
    for i in "${!workers[@]}"; do
        CUDA_VISIBLE_DEVICES="$i" "$py" "$run/$script" --worker "${workers[$i]}" > "/workspace/run058-control/$label-${workers[$i]}.log" 2>&1 &
        pids+=("$!")
        printf '%s %s %s\n' "$label" "${workers[$i]}" "$!" >> /workspace/run058-control/processes.txt
    done
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    [[ "$failed" == 0 ]]
}
phase 05_remote_preflight.py preflight
"$py" "$run/01_validate_preflights.py"
touch /workspace/run058-control/preflight-passed
phase 02_train.py training
"$py" "$run/03_verify.py"
touch /workspace/run058-control/completed
