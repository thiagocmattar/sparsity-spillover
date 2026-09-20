ps -eo pid,ppid,comm,pcpu,etime --sort=-pcpu | head -n 16
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader
python3 - <<'PY'
from pathlib import Path
import time,json
root=Path('/workspace/run041-latency/runtime')
paths=[p for parent in ['extensions','triton'] for p in (root/parent).rglob('*') if p.is_file()]
recent=sorted(paths,key=lambda p:p.stat().st_mtime,reverse=True)[:8]
print(json.dumps([{'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'age':time.time()-p.stat().st_mtime} for p in recent]))
p=Path('/workspace/run041-latency/artifacts/smoke/smoke-c00-r1-001.log')
with p.open('rb') as f:
 f.seek(max(0,p.stat().st_size-1200));print(f.read().decode(errors='replace'))
PY
