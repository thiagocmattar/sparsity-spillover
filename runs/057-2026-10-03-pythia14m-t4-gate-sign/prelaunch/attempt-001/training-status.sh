set -eu
date -u
python3 - <<'PY'
import json,statistics,time
from pathlib import Path
run=Path('/workspace/sparsity-spillover/runs/057-2026-10-03-pythia14m-t4-gate-sign')
pre=[]
for p in sorted((run/'prelaunch').glob('remote-preflight-*.json')):
    r=json.loads(p.read_text())
    pre.append({'worker':r['worker'],'status':r['status'],'seconds':r.get('median_boundary_seconds'),
                'memory_fraction':r.get('peak_reserved_bytes',0)/r.get('total_memory_bytes',1)})
print(json.dumps({'preflights':pre}))
for p in sorted((run/'artifacts/attempts').glob('*/manifest.json')):
    m=json.loads(p.read_text());events=p.parent/'events.jsonl';rows=[]
    for line in events.read_text().splitlines():
        try:r=json.loads(line)
        except json.JSONDecodeError:continue
        if r.get('event')=='train':rows.append(r)
    info={'condition':m['condition']['id'],'status':m['status']}
    if rows:
        r=rows[-1];seconds=statistics.median(t['step_wall_seconds'] for t in rows[-20:])
        info.update(step=r['step'],task_loss=r['task_loss'],pressure_loss=r['pressure_loss'],
            tokens_per_second=2097152/seconds,training_remaining_seconds=(712-r['step'])*seconds,
            event_age_seconds=time.time()-events.stat().st_mtime,skipped=sum(t['optimizer_step_skipped'] for t in rows),
            reserved_bytes=r['peak_gpu_memory_reserved_bytes'])
    metrics=p.parent/'metrics.json'
    if metrics.exists():
        final=json.loads(metrics.read_text())
        info.update(final_validation_loss=final['validation']['final']['loss'],
                    total_seconds=final['timing']['total_seconds'])
    print(json.dumps(info))
verification=run/'artifacts/verification.json'
if verification.exists():print(json.dumps({'cohort_verification':json.loads(verification.read_text())}))
PY
test ! -f /workspace/run057-control/pipeline.exit || cat /workspace/run057-control/pipeline.exit
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
