set -euo pipefail
for attempt in $(seq 1 60); do
  if test -f /workspace/run054-control/environment-ready; then break; fi
  sleep 5
done
test -f /workspace/run054-control/environment-ready
export HF_HOME=/workspace/run054-hf-cache RAYON_NUM_THREADS=24 TOKENIZERS_PARALLELISM=true
cd /workspace/sparsity-spillover
/workspace/run054-venv/bin/python runs/004-2026-08-29-pythia14m-full-pass-l1n/06_build_cache_from_hf.py --batch-size 1024 --log-every-documents 10000
touch /workspace/run054-control/cache-ready
