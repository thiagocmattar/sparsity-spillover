#!/usr/bin/env bash
set -euo pipefail
cd /workspace/run067
source /workspace/run067-env/bin/activate
export PYTHONPATH=/workspace/run067/src
export MAX_JOBS=2
export HF_HUB_OFFLINE=1
: "${RUN067_SOURCE_COMMIT:?Verified deployment commit required}"
run=runs/067-2026-10-08-pythia14m-sparse-vocabulary-kernel
mkdir -p "$run/prelaunch"
cp /workspace/run067-environment.txt "$run/prelaunch/cloud-environment.txt"
python -u "$run/05_verify.py" --inputs-only
python -u "$run/00_component.py" --tag cloud001
for model in t2-tile-head-ph-kappa-0 t2-tile-head-ph-kappa-0p1 t2-tile-head-ph_head-kappa-0 t2-tile-head-ph_head-kappa-0p1; do
    python -u "$run/02_benchmark.py" --model "$model" --replicate 1 --attempt "smoke-$model-001" --smoke
done
for replicate in 1 2 3; do
    for model in t2-tile-head-ph-kappa-0 t2-tile-head-ph-kappa-0p1 t2-tile-head-ph_head-kappa-0 t2-tile-head-ph_head-kappa-0p1; do
        python -u "$run/02_benchmark.py" --model "$model" --replicate "$replicate" --attempt "full-$model-$replicate-001"
    done
done
python -u "$run/05_verify.py" --tag 001
python -u "$run/06_seal.py" --tag 001
