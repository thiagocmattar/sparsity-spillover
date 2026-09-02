#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 GPU_TYPE_ID CLOUD_TYPE HOURLY_PRICE_USD SLUG" >&2
    exit 64
fi

gpu_type_id=$1
cloud_type=$2
hourly_price_usd=$3
slug=$4
repo=/workspace/sparsity-spillover
control=/workspace/run019-control
run_dir="$repo/runs/019-2026-09-01-pythia410m-selected-ladder-canonical-init"
python=/workspace/run019-venv/bin/python
output="$control/calibration-$slug.json"

test "$(cat "$control/setup-exit-code.txt")" = 0
test ! -e "$output"
date -u +%FT%TZ > "$control/calibration-$slug-started-utc.txt"
{
    date -u +%FT%TZ
    uname -a
    "$python" --version
    "$python" -c 'import torch, transformers; print(f"torch={torch.__version__} cuda={torch.version.cuda} transformers={transformers.__version__}")'
    nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader
    git -C "$repo" rev-parse HEAD
} > "$control/execution-context-$slug.txt" 2>&1

status=0
{
    set -euo pipefail
    cd "$repo"
    "$python" "$run_dir/05_calibrate.py" \
        --gpu-type-id "$gpu_type_id" \
        --cloud-type "$cloud_type" \
        --hourly-price-usd "$hourly_price_usd" \
        --output "$output"
} > "$control/calibration-$slug.log" 2>&1 || status=$?

printf '%s\n' "$status" > "$control/calibration-$slug-exit-code.txt"
date -u +%FT%TZ > "$control/calibration-$slug-finished-utc.txt"
nvidia-smi -q > "$control/nvidia-smi-q-after-$slug.txt" 2>&1 || true
exit "$status"
