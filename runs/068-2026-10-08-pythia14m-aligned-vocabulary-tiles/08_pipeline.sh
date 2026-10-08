#!/usr/bin/env bash
# One independent condition per GPU. Preflight state is discarded before training.
set -euo pipefail
repo=/workspace/sparsity-spillover
run="$repo/runs/068-2026-10-08-pythia14m-aligned-vocabulary-tiles"
py=/workspace/run068-venv/bin/python
cd "$repo"
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
mkdir -p "$run/prelaunch" /workspace/run068-control
mapfile -t workers < <("$py" -c "import sys;sys.path.insert(0,'$run');from run_config import load_config;print('\n'.join(load_config()['runpod']['worker_assignments']))")
if [[ ${#workers[@]} != 2 ]]; then echo 'Expected two workers' >&2; exit 1; fi
gpu_count=$("$py" -c "import sys;sys.path.insert(0,'$run');from run_config import load_config;print(load_config()['runpod']['gpu_count'])")
phase() {
    local script="$1" label="$2" failed=0 start i slot
    local -a pids=()
    for ((start=0; start<${#workers[@]}; start+=gpu_count)); do
        pids=()
        for ((i=start; i<start+gpu_count && i<${#workers[@]}; i++)); do
            slot=$((i-start))
            CUDA_VISIBLE_DEVICES="$slot" "$py" "$run/$script" --worker "${workers[$i]}" > "/workspace/run068-control/$label-${workers[$i]}.log" 2>&1 &
            pids+=("$!")
            printf '%s %s %s\n' "$label" "${workers[$i]}" "$!" >> /workspace/run068-control/processes.txt
        done
        for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
        [[ "$failed" == 0 ]] || return 1
    done
}
phase 05_remote_preflight.py preflight
"$py" "$run/01_validate_preflights.py"
touch /workspace/run068-control/preflight-passed
phase 02_train.py training
"$py" "$run/03_verify.py"
touch /workspace/run068-control/completed
