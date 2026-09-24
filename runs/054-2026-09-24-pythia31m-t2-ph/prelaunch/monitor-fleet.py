"""Read one bounded fleet snapshot; no remote mutations."""
from concurrent.futures import ThreadPoolExecutor
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess

RUN=Path(__file__).resolve().parent.parent
NODES=json.loads((RUN/'prelaunch/parallel-connections-002.json').read_text())
KEY=str(Path.home()/'.runpod/ssh/runpodctl-ssh-key')
SCRIPT='''import json,pathlib,statistics,time,math,subprocess
r=pathlib.Path('/workspace/sparsity-spillover/runs/054-2026-09-24-pythia31m-t2-ph')
out={'conditions':[],'pipelines':[]}
for p in sorted((r/'artifacts/pipeline').glob('*/status.json')):
 try:out['pipelines'].append(json.loads(p.read_text()))
 except (ValueError,OSError):pass
for a in sorted((r/'artifacts/attempts').glob('*')):
 try: manifest=json.loads((a/'manifest.json').read_text())
 except (ValueError,OSError):continue
 events=[]
 for line in (a/'events.jsonl').read_text().splitlines():
  try:
   e=json.loads(line)
   if e.get('event')=='train':events.append(e)
  except ValueError:pass
 if events:
  e=events[-1];sec=statistics.median(x['step_wall_seconds'] for x in events[-20:])
  out['conditions'].append({'id':manifest['condition']['id'],'attempt':a.name,'status':manifest['status'],
   'step':e['step'],'loss':e['task_loss'],'tokens_per_second':2097152/sec,'remaining_minutes':max(0,712-e['step'])*sec/60,
   'skipped_steps':sum(bool(x.get('optimizer_step_skipped')) for x in events),
   'stale_seconds':time.time()-(a/'events.jsonl').stat().st_mtime,'all_losses_finite':all(math.isfinite(x['task_loss']) for x in events)})
out['gpus']=subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout.strip()
out['disk']=subprocess.run(['df','-h','/workspace'],capture_output=True,text=True).stdout.strip().splitlines()[-1]
out['kernel_ready']=pathlib.Path('/workspace/run054-control/kernel-ready').exists()
out['latency']=[]
for p in sorted((r/'latency/artifacts/attempts').glob('final-*')):
 try:
  m=json.loads((p/'manifest.json').read_text());events=(p/'events.jsonl').read_text().splitlines()
  out['latency'].append(dict(attempt=p.name,status=m['status'],qualified=m.get('qualified'),
   latest=json.loads(events[-1]) if events else None,timing=m.get('timing')))
 except (ValueError,OSError):pass
print(json.dumps(out))
'''


def read(node):
    s=node['ssh']
    cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-i',KEY,'-p',str(s['port']),s['username']+'@'+s['host'],'python3 -']
    try:
        p=subprocess.run(cmd,input=SCRIPT,text=True,capture_output=True,timeout=25,check=True)
        return dict(pod=node['name'],**json.loads(p.stdout))
    except Exception as error:
        return dict(pod=node['name'],error=str(error))


def main():
    p=argparse.ArgumentParser();p.add_argument('--latency-only',action='store_true');args=p.parse_args()
    retired={json.loads(path.read_text())['pod'] for path in (RUN/'prelaunch').glob('terminated-*.json')
        if json.loads(path.read_text())['status']=='terminated'}
    nodes=[node for node in (NODES[4:] if args.latency_only else NODES) if node['id'] not in retired]
    with ThreadPoolExecutor(max_workers=5) as pool:
        rows=list(pool.map(read,nodes))
    stamp=datetime.now(timezone.utc)
    folder=RUN/'prelaunch/monitoring';folder.mkdir(exist_ok=True)
    name=('latency-' if args.latency_only else '')+stamp.strftime('%Y%m%d-%H%M%S')+'.json'
    (folder/name).write_text(json.dumps(dict(utc=stamp.isoformat(),pods=rows),indent=2)+'\n')
    for row in rows:
        if 'error' in row:print(json.dumps(row),flush=True);continue
        summary=dict(pod=row['pod'],training=[dict(condition=c['id'],step=c['step'],loss=round(c['loss'],6),
            tokens_per_second=round(c['tokens_per_second']),remaining_minutes=round(c['remaining_minutes'],1),
            status=c['status'],skipped_steps=c['skipped_steps'],finite=c['all_losses_finite']) for c in row['conditions']],
            pipelines=[dict(status=p['status'],workers={k:v['stage'] for k,v in p['workers'].items()},error=p.get('error')) for p in row['pipelines']],
            latency=[dict(attempt=l['attempt'],status=l['status'],qualified=l['qualified'],
                stage=(l['latest'] or {}).get('stage'),
                ms={k:round(v['geomean_host_ms'],6) for k,v in (l['timing'] or {}).items()}) for l in row['latency']])
        print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
