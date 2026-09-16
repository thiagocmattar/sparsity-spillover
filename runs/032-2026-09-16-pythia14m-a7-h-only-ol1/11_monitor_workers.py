"""One read-only snapshot of the five Run 032 Pods; no polling loop."""
import concurrent.futures
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("run032_remote", HERE / "07_remote.py")
remote = importlib.util.module_from_spec(spec)
spec.loader.exec_module(remote)

SCRIPT = r'''python3 - <<'PY'
from datetime import datetime, timezone
import json, os, pathlib, shutil, subprocess, time
w=pathlib.Path('/workspace')
r=w/'sparsity-spillover/runs/032-2026-09-16-pythia14m-a7-h-only-ol1'
out={'utc':datetime.now(timezone.utc).isoformat(),'free_workspace_bytes':shutil.disk_usage(w).free}
for phase in ['environment','cache','ready']:
    name='run032-ready' if phase=='ready' else 'run032-'+phase+'-ready'
    out[phase+'_ready']=(w/name).exists()
for phase in ['preflight','training']:
    f=w/('run032-'+phase+'.exit')
    if f.exists(): out[phase+'_exit']=f.read_text().strip()
    f=w/('run032-'+phase+'.pid')
    if f.exists():
        pid=int(f.read_text())
        try: os.kill(pid,0); alive=True
        except ProcessLookupError: alive=False
        out[phase+'_alive']=alive
f=r/'prelaunch/remote-preflight.json'
if f.exists():out['preflight_status']=json.loads(f.read_text())['status']
out['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader'],text=True).strip()
out['attempts']=[]
for a in sorted((r/'artifacts/attempts').glob('*')):
    if not a.is_dir():continue
    row={'attempt':a.name}
    f=a/'manifest.json'
    if f.exists():row['status']=json.loads(f.read_text()).get('status')
    f=a/'events.jsonl'
    if f.exists():
        events=[]
        for line in f.read_text().splitlines():
            try:events.append(json.loads(line))
            except json.JSONDecodeError:pass
        trains=[e for e in events if e.get('event')=='train']
        row['event_age_seconds']=round(time.time()-f.stat().st_mtime,1)
        if events:row['last_event']=events[-1]['event']
        if trains:
            t=trains[-1]
            keys=['step','elapsed_seconds','task_loss','pressure_loss','tokens_per_second','optimizer_step_skipped','peak_gpu_memory_reserved_bytes','loss_scale','pressure_to_task_ratio_final','trust_scale','pressure_capture_tensor_count']
            row.update({k:t[k] for k in keys if k in t})
            if len(trains)>1:
                old=trains[max(0,len(trains)-51)]
                seconds=(t['elapsed_seconds']-old['elapsed_seconds'])/(t['step']-old['step'])
                row['recent_end_to_end_seconds_per_step']=seconds
                row['recent_end_to_end_tokens_per_second']=2097152/seconds
                row['training_remaining_seconds']=(712-t['step'])*seconds
        row['skipped_updates']=sum(bool(e.get('optimizer_step_skipped')) for e in trains)
    out['attempts'].append(row)
print(json.dumps(out))
PY'''


def snapshot(pod):
    try:
        with remote.connect(pod['id']) as client:
            value=json.loads(remote.command(client,SCRIPT))
        return {'pod':pod['id'],'condition':pod['condition'],**value}
    except Exception as error:
        return {'pod':pod['id'],'condition':pod['condition'],'error':str(error)}


if __name__=='__main__':
    pods=json.loads((HERE/'prelaunch/pods.json').read_text(encoding='utf-8-sig'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        rows=list(pool.map(snapshot,pods))
    result={'utc':datetime.now(timezone.utc).isoformat(),'workers':rows}
    with (HERE/'prelaunch/monitoring.jsonl').open('a',encoding='utf-8') as handle:
        handle.write(json.dumps(result)+'\n')
    brief=[]
    for row in rows:
        item={k:row[k] for k in ['condition','error','training_exit','training_alive','preflight_status','gpu'] if k in row}
        for attempt in row.get('attempts',[]):
            item.update({k:attempt[k] for k in ['status','step','task_loss','pressure_loss','skipped_updates','pressure_capture_tensor_count','pressure_to_task_ratio_final','loss_scale','event_age_seconds'] if k in attempt})
            if 'recent_end_to_end_tokens_per_second' in attempt:
                item['tokens_per_second']=round(attempt['recent_end_to_end_tokens_per_second'])
                item['training_remaining_minutes']=round(attempt['training_remaining_seconds']/60,1)
        brief.append(item)
    print(json.dumps({'utc':result['utc'],'workers':brief}))
