python3 - <<'PY'
import json,subprocess
from pathlib import Path
c=Path('/workspace/run066-control')
r=Path('/workspace/sparsity-spillover/runs/066-2026-10-08-pythia14m-vocabulary-tile-gates')
status={'stage_markers':{n:(c/n).exists() for n in ['environment-ready','cache-ready','preflight-passed','completed']}}
status['pipeline_exit']=(c/'pipeline.exit').read_text().strip() if (c/'pipeline.exit').exists() else None
status['seal_exit']=(r/'transfer/seal.exit').read_text().strip() if (r/'transfer/seal.exit').exists() else None
status['sealed_archive_receipt']=(r/'transfer/training-receipt-001.json').exists()
status['logs']={p.name:p.read_text(errors='replace').splitlines()[-3:] for p in sorted(c.glob('*.log')) if p.name in ('guard.log','pipeline.log') or 'Traceback' in p.read_text(errors='replace')}
status['preflights']=[]
for p in sorted((r/'prelaunch/attempts/002').glob('remote-preflight-*.json')):
    try:
        x=json.loads(p.read_text());status['preflights'].append({k:x.get(k) for k in ['worker','status','median_boundary_seconds','peak_reserved_bytes','total_memory_bytes','checkpoint_numerical_smoke']})
    except json.JSONDecodeError:pass
status['gpu']=subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
status['disk']=subprocess.run(['df','-h','/workspace'],capture_output=True,text=True).stdout.strip()
status['owned_workspace_bytes']=subprocess.run(['du','-sb','/workspace/sparsity-spillover','/workspace/run066-venv'],capture_output=True,text=True).stdout.strip()
py=Path('/workspace/run066-venv/bin/python')
if (c/'environment-ready').exists() and (r/'artifacts/attempts').exists():
    result=subprocess.run([str(py),str(r/'04_monitor.py')],capture_output=True,text=True)
    status['training']=json.loads(result.stdout) if result.returncode==0 else {'monitor_error':result.stderr[-1000:]}
    if result.returncode==0:
        for row in status['training']:
            events=r/'artifacts/attempts'/row['attempt']/'events.jsonl'
            for line in events.read_text().splitlines():
                try: event=json.loads(line)
                except json.JSONDecodeError: continue
                if event.get('worker_id'): row['worker']=event['worker_id']
                if event.get('event')=='validation': row['latest_validation']=event
            base=events.parent/'diagnostics'
            paths=sorted(base.glob('step_*/activation_statistics.json'))
            if (base/'activation_statistics.json').exists(): paths.append(base/'activation_statistics.json')
            if paths:
                try:
                    stats=json.loads(paths[-1].read_text())
                    head=next(x for x in stats['rows'] if x['name']=='head_in.final')
                    tiles=next(x for x in stats['occupancy_rows'] if x['name']=='head_in.final')['tiles']['8x16']
                    row['head_diagnostic']={'source':paths[-1].relative_to(r).as_posix(),'exact_zero_count':head['exact_zero_count'],'elements':head['total'],'empty_8x16_tiles':tiles['empty'],'total_8x16_tiles':tiles['total']}
                except json.JSONDecodeError: pass
print(json.dumps(status,indent=2))
PY
