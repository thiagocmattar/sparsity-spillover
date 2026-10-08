"""One bounded read-only status check of the owned profiling job."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from runio import module
cloud=module('run065_cloud',ROOT/'06_cloud.py');client,_=cloud.connect()
try:
    command="""python3 - <<'PY'
from pathlib import Path
import json,subprocess,sys
root=Path('/workspace/run065');sys.path.insert(0,str(root))
from trace_copy_mapping import map_native_copies
states=[{'attempt':p.parent.name,**json.loads(p.read_text())} for p in (root/'artifacts/attempts').glob('*/status.json')]
control=Path('/workspace/run065-control/pipeline.exit')
out={'pipeline_exit':control.read_text().strip() if control.exists() else None,'smokes_complete':sum(r['attempt'].startswith('smoke') and r['stage']=='complete' for r in states),'full_complete':sum(r['attempt'].startswith('full') and r['stage']=='complete' for r in states),'active_or_failed':[r for r in states if r['stage']!='complete'],'last_full_losses':[{'attempt':r['attempt'],'loss':r['loss']} for r in states if r['attempt'].startswith('full') and r['stage']=='complete'][-2:]}
out['trace_mappings']=[]
for p in (root/'artifacts/attempts').glob('full-*/traces.json'):
 rows=json.loads(p.read_text());mapped=[map_native_copies(r) for r in rows]
 out['trace_mappings'].append({'attempt':p.parent.name,'mapped':sum(m['mapped'] for m in mapped),'total':len(rows),'failures':[{'mode':r['mode'],'input':r['input_index'],'reason':m['reason']} for r,m in zip(rows,mapped) if not m['mapped']]})
out['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],text=True).strip()
out['disk']=subprocess.check_output(['df','-h','/workspace','/opt'],text=True).strip()
print(json.dumps(out))
PY"""
    print(cloud.execute(client,command))
finally:client.close()
