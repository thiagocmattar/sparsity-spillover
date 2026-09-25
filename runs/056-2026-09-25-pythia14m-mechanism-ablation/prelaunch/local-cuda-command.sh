set -euo pipefail
source /opt/sparsity-gpu/activate.sh
cd /home/researcher/sparsity-spillover/run056-development-001
mkdir -p prelaunch
export MAX_JOBS=2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TORCH_EXTENSIONS_DIR=/home/researcher/.cache/torch_extensions/run056-development
/opt/sparsity-gpu/venv/bin/python 01_prepare.py verify
set +e
timeout --signal=TERM --kill-after=10s 600s /opt/sparsity-gpu/venv/bin/python -u 06_cuda_checks.py --development >prelaunch/cuda-development.log 2>&1
code=$?
echo "$code" >prelaunch/cuda-development.exit
exit "$code"
