set -euo pipefail
cd /workspace/sparsity-spillover
run=runs/043-2026-09-20-pythia70m-hz-h-only-ol1
control=/workspace/run043-control
export PYTHONPATH="$PWD/$run:$PWD/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HOME=/workspace/run043-hf
workers=(hz-h-ol1-kappa-0 hz-h-ol1-kappa-0p01 hz-h-ol1-kappa-0p05 hz-h-ol1-kappa-0p1)
pids=()
for i in 0 1 2 3; do
 CUDA_VISIBLE_DEVICES=$i /workspace/run043-venv/bin/python -u "$control/thread-calibration-002.py" --worker "${workers[$i]}" > "$control/thread-calibration-002-$i.log" 2>&1 &
 pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
printf '%s\n' "$failed" > "$control/thread-calibration-002.exit"
exit "$failed"
