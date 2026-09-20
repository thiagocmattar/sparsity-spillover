python3 - <<'PY'
from pathlib import Path
import json,time,shutil
root=Path('/workspace/sparsity-spillover/runs/044-2026-09-20-pythia14m-hz-h-only-ol1-kappa05')
control=Path('/workspace/run044-control')
result={'epoch':time.time(),'workers':[],'free_disk_GiB':shutil.disk_usage('/workspace').free/1024**3}
for path in sorted((root/'artifacts/attempts').glob('*/events.jsonl')):
    events=[]
    for line in path.read_text().splitlines():
        try:events.append(json.loads(line))
        except ValueError:pass
    trains=[e for e in events if e.get('event')=='train']
    if not trains:continue
    last=trains[-1];first=trains[max(0,len(trains)-31)]
    seconds=(last['elapsed_seconds']-first['elapsed_seconds'])/(last['step']-first['step']) if last['step']>first['step'] else last['elapsed_seconds']
    result['workers'].append({'condition':last['condition_id'],'step':last['step'],'loss':last['task_loss'],
        'recent_step_seconds':seconds,'tokens_per_second':2097152/seconds,
        'remaining_training_seconds':(712-last['step'])*seconds,'event_age_seconds':time.time()-path.stat().st_mtime,
        'overflow_count':sum(e['gradient_overflow'] for e in trains),'last_phase':events[-1]['event'],
        'reserved_GiB':last['peak_gpu_memory_reserved_bytes']/1024**3})
result['pipeline_exit']=(control/'pipeline.exit').read_text() if (control/'pipeline.exit').exists() else None
result['verified']=(control/'training-verified').exists()
result['sealed']=(control/'training-sealed').exists()
print(json.dumps(result,indent=2))
PY
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
