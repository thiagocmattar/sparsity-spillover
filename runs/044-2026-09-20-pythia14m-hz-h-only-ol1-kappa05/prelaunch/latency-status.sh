set -eu
date -u
control=/workspace/run044-control
root=/workspace/run044-latency
for file in environment.exit compile-002.exit compile-003.exit compile-004.exit pipeline.exit; do
  if test -e "$control/$file"; then printf '%s: ' "$file"; cat "$control/$file"; fi
done
for file in smoke scientific collection; do
  if test -e "$control/$file.log"; then printf '\n%s\n' "$file"; tail -c 1400 "$control/$file.log"; fi
done
python3 - <<'PY'
from pathlib import Path
import json,time
root=Path('/workspace/run044-latency/artifacts')
logs=sorted(root.glob('*/*.log'),key=lambda p:p.stat().st_mtime)
if logs:
    path=logs[-1]
    print(json.dumps({'latest_leaf_log':str(path),'age_seconds':time.time()-path.stat().st_mtime}))
    print(path.read_text(errors='replace')[-1800:])
for path in sorted(root.glob('attempts/*/result.json')):
    result=json.loads(path.read_text())
    print(json.dumps({'attempt':path.parent.name,'status':result.get('status'),'qualified':result.get('qualified'),'loss':result.get('loss')}))
PY
for file in environment-ready precompile-ready precompile-extra-ready; do
  if test -e "$root/runtime/$file"; then echo "$file"; fi
done
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader
ps -C nvcc,cc1plus,cicc,ptxas,ninja -o pid,etime,pcpu,rss,comm || true
ps -C python,python3 -o pid,etime,pcpu,rss,args | tail -n 8
