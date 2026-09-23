"""Read-only status snapshot for this Run052 Pod."""
import json,time,subprocess,shutil
from pathlib import Path

run=Path('/workspace/run052/052-2026-09-23-pythia70m-sparse-attribution')
rows=[]
for path in sorted((run/'artifacts').glob('*/status.json'),key=lambda p:p.stat().st_mtime)[-4:]:
    value=json.loads(path.read_text());row={'attempt':path.parent.name,'age_seconds':round(time.time()-path.stat().st_mtime,1)}
    for key in ('utc','stage','condition','implementation','index','total','blocks','target_blocks','blocks_per_second','remaining_seconds','elapsed_seconds','error','command','qualified'):
        if key in value:row[key]=value[key]
    if 'loss' in value:row['native_loss']=value['loss'].get('native') if isinstance(value['loss'],dict) else value['loss']
    rows.append(row)
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu','--format=csv,noheader'],text=True).strip()
print(json.dumps({'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'status':rows,'gpu_MiB_used_free_util_temp':gpu,'container_free_GB':round(shutil.disk_usage('/').free/1e9,2)},indent=2))
