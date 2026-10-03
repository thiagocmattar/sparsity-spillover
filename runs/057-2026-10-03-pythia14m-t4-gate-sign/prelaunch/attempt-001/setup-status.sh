set -eu
date -u
tail -n 12 /workspace/run057-control/pipeline.log
if test -f /workspace/run057-control/cache.log; then tail -n 5 /workspace/run057-control/cache.log; fi
if test -f /workspace/run057-control/pipeline.exit; then cat /workspace/run057-control/pipeline.exit; fi
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
for split in train validation; do
    stat --format='%n %s bytes' /workspace/sparsity-spillover/data/tokenized/minipile-pythia-14m-full/$split/tokens.int32.bin 2>/dev/null || true
done
