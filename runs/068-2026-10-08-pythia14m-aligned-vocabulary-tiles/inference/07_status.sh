#!/usr/bin/env bash
set -eu
date -u '+STATUS %Y-%m-%dT%H:%M:%SZ'
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader
python3 - <<'PY'
import json,time
from pathlib import Path
from statistics import median
root=Path('/workspace/run068-head/runs/068-2026-10-08-pythia14m-aligned-vocabulary-tiles/inference')
complete=[]
for p in sorted((root/'artifacts/attempts').glob('full-*/result.json')):
 row=json.loads(p.read_text())
 complete.append({'attempt':p.parent.name,'status':row['status'],'elapsed_seconds':row['elapsed_seconds'],'loss':row.get('loss')})
paths=sorted((root/'artifacts/attempts').glob('*/status.json'),key=lambda p:p.stat().st_mtime)
current=None
if paths:
 p=paths[-1];current={'attempt':p.parent.name,'status':json.loads(p.read_text()),'stale_seconds':time.time()-p.stat().st_mtime}
passed=[r for r in complete if r['status']=='complete']
eta=None
if passed:
 eta=(6-len(passed))*median(r['elapsed_seconds'] for r in passed)
 if current and current['status']['stage']!='complete':
  eta=max(0,eta-current['status']['elapsed_seconds'])
print(json.dumps({'complete_full_processes':len(passed),'expected_full_processes':6,'processes':complete,
                  'current':current,'estimated_remaining_process_seconds':eta,
                  'etc_scope':'fresh full processes only; excludes setup/sealing/transfer'},indent=2))
control=Path('/workspace/run068-control')
if (control/'pipeline.exit').exists():print('PIPELINE_EXIT', (control/'pipeline.exit').read_text().strip())
if (control/'pipeline.log').exists():
 print('PIPELINE_LOG_TAIL')
 print('\n'.join((control/'pipeline.log').read_text(errors='replace').splitlines()[-5:]))
PY
