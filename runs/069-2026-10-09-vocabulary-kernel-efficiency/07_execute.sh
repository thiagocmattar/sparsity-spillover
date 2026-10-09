#!/usr/bin/env bash
set -euo pipefail
cd /workspace/run069
source /workspace/run069-env/bin/activate
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export NCU="$(cat /workspace/run069-ncu-path.txt)"
export MAX_JOBS=2
export TORCH_EXTENSIONS_DIR=/workspace/run069-extensions
export HF_HUB_OFFLINE=1
export PYTHONPATH=/workspace/run069/src
: "${RUN069_SOURCE_COMMIT:?Verified deployment commit required}"
run=runs/069-2026-10-09-vocabulary-kernel-efficiency
mkdir -p "$run/artifacts/control"
cp /workspace/run069-environment.txt "$run/artifacts/control/environment.txt"
trap 'code=$?; printf "%s\n" "$code" > "$run/artifacts/control/pipeline.exit"' EXIT
python -u "$run/02_component.py" --tag cloud
python -u "$run/03_capture.py"
for model in t2-hz0-head16-ph_head-kappa-0p1 t2-hz0-head16-ph_head-kappa-0p5; do
    python -u "$run/04_screen.py" --model "$model"
done
python -u "$run/profile_driver.py"
for replicate in 1 2 3; do
    for model in t2-hz0-head16-ph_head-kappa-0p1 t2-hz0-head16-ph_head-kappa-0p5; do
        python -u "$run/05_endpoint.py" --model "$model" --replicate "$replicate"
    done
done
python -u "$run/08_verify.py"
python -u "$run/11_reduce.py"
python -u "$run/08_verify.py"
