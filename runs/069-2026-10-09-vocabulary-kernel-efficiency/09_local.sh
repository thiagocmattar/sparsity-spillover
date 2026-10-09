#!/usr/bin/env bash
set -euo pipefail
source /opt/sparsity-gpu/activate.sh
export MAX_JOBS=2
export HF_HUB_OFFLINE=1
run=runs/069-2026-10-09-vocabulary-kernel-efficiency
cd "${1:?isolated checkout root required}"
mkdir -p "$run/prelaunch"
exec > "$run/prelaunch/local-cuda.txt" 2>&1
python -u "$run/02_component.py" --tag local
python -u "$run/03_capture.py" --development --inputs 2
for model in t2-hz0-head16-ph_head-kappa-0p1 t2-hz0-head16-ph_head-kappa-0p5; do
    python -u "$run/04_screen.py" --model "$model" --development
    python -u "$run/05_endpoint.py" --model "$model" --replicate 1 --development --blocks 2
done
export NCU=/opt/nvidia/nsight-compute/2025.1.1/ncu
python -u "$run/profile_driver.py" --development
