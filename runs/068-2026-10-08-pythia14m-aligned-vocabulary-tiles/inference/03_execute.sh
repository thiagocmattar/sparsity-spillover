#!/usr/bin/env bash
set -euo pipefail
cd /workspace/run068-head
source /workspace/run068-head-env/bin/activate
export PYTHONPATH=/workspace/run068-head/src
export MAX_JOBS=2
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
test -x "$CUDA_HOME/bin/nvcc"
export HF_HUB_OFFLINE=1
: "${RUN068_HEAD_SOURCE_COMMIT:?Verified deployment commit required}"
run=runs/068-2026-10-08-pythia14m-aligned-vocabulary-tiles/inference
mkdir -p "$run/prelaunch"
cp /workspace/run068-head-environment.txt "$run/prelaunch/cloud-environment.txt"
python -u "$run/05_verify.py" --inputs-only
python -u "$run/00_component.py" --tag cloud001
for model in t2-hz0-head16-ph_head-kappa-0p1 t2-hz0-head16-ph_head-kappa-0p5; do
    python -u "$run/02_benchmark.py" --model "$model" --replicate 1 --attempt "smoke-$model-001" --smoke
done
for replicate in 1 2 3; do
    for model in t2-hz0-head16-ph_head-kappa-0p1 t2-hz0-head16-ph_head-kappa-0p5; do
        python -u "$run/02_benchmark.py" --model "$model" --replicate "$replicate" --attempt "full-$model-$replicate-001"
    done
done
python -u "$run/05_verify.py" --tag 001
python -u "$run/06_seal.py" --tag 001
