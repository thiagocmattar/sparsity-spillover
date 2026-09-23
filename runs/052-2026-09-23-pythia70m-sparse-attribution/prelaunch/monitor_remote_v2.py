"""Read-only Run052 status plus the latest completed numerical comparison."""
import json,time,subprocess,shutil
from pathlib import Path

run=Path('/workspace/run052/052-2026-09-23-pythia70m-sparse-attribution')
paths=sorted((run/'artifacts').glob('*/status.json'),key=lambda p:p.stat().st_mtime)
rows=[]
for path in paths[-2:]:
    value=json.loads(path.read_text());row={'attempt':path.parent.name,'age_seconds':round(time.time()-path.stat().st_mtime,1)}
    for key in ('utc','stage','condition','implementation','index','total','blocks','target_blocks','blocks_per_second','remaining_seconds','elapsed_seconds','error','qualified'):
        if key in value:row[key]=value[key]
    if 'loss' in value:row['native_loss']=value['loss'].get('native') if isinstance(value['loss'],dict) else value['loss']
    rows.append(row)
completed=[]
for path in (run/'artifacts').glob('*/result.json'):
    value=json.loads(path.read_text())
    if value.get('status')=='complete' and 'qualification' in value:
        completed.append((path.stat().st_mtime,path,value))
last=None
if completed:
    _,path,value=max(completed,key=lambda row:row[0])
    last={'attempt':path.parent.name,'blocks':value['validation_blocks'],'native_loss':value['loss']['native'],
          'all_qualified':all(value['qualification'].values()),'failed_modes':[k for k,v in value['qualification'].items() if not v],
          'process_seconds':value['elapsed_seconds']}
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu','--format=csv,noheader'],text=True).strip()
print(json.dumps({'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'status':rows,'last_completed':last,
                  'gpu_MiB_used_free_util_temp':gpu,'container_free_GB':round(shutil.disk_usage('/').free/1e9,2)},indent=2))
