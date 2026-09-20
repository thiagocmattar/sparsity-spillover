set -eu
control=/workspace/run044-control
date -u
stat -c '%s %n' /workspace/sparsity-spillover/runs/044-2026-09-20-pythia14m-hz-h-only-ol1-kappa05/inputs/random-initialization/model.safetensors* 2>/dev/null || true
for name in pipeline cache preflight-0 training-0 verification sealing; do
  if test -f "$control/$name.log"; then
    printf '\n%s\n' "$name"
    tail -c 1000 "$control/$name.log"
  fi
done
if test ! -e "$control/environment-ready"; then tail -c 1200 "$control/setup.log"; fi
if test -f "$control/pipeline.exit"; then cat "$control/pipeline.exit"; fi
df -h /workspace
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
