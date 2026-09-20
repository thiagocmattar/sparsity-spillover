set -euo pipefail
control=/workspace/run043-control
test "$(cat "$control/pipeline.exit")" = 1
test ! -e "$control/preflight-passed"
test ! -e "$control/thread-calibration-002.sh"
cp "$control/pipeline.exit" "$control/pipeline-001.exit"
cp "$control/pipeline.log" "$control/pipeline-001.log"
cat > "$control/thread-calibration-002.py" <<'PY'
import importlib.util,json
from pathlib import Path
root=Path('/workspace/sparsity-spillover/runs/043-2026-09-20-pythia70m-hz-h-only-ol1')
spec=importlib.util.spec_from_file_location('calibration',root/'05_remote_preflight.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.RUN_DIR=root/'prelaunch/thread-calibration-002'
module.main()
PY
cat > "$control/thread-calibration-002.sh" <<'SH'
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
SH
nohup setsid bash "$control/thread-calibration-002.sh" > "$control/thread-calibration-002.log" 2>&1 < /dev/null &
echo $! > "$control/pipeline.pid"
printf 'Detached four-GPU OMP1 calibration started; original OMP8 outputs retained.\n'
