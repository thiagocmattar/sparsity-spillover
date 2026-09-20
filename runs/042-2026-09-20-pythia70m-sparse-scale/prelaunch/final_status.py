"""One bounded read-only status query for the frozen qualification pipeline."""
import json
from pathlib import Path
import remote


COMMAND=r'''cd /workspace/run042 && python3 - <<'PY'
import json,time,statistics,subprocess
from pathlib import Path
rows={}
for phase in ('final','decomposition'):
 p=Path('artifacts')/phase/'summary-024.json'
 data=json.loads(p.read_text()) if p.exists() else []
 rows[phase]={'completed':len(data),'qualified':sum(bool(r.get('qualified')) for r in data),
  'median_seconds':statistics.median(r['seconds'] for r in data) if data else None,
  'last':data[-1] if data else None}
status=sorted((p for p in Path('artifacts/attempts').glob('*/status.json') if p.parent.name.endswith('-024')),key=lambda p:p.stat().st_mtime)
latest=None
if status:
 p=status[-1];latest={'attempt':p.parent.name,**json.loads(p.read_text())}
 events=[json.loads(x) for x in p.with_name('events.jsonl').read_text().splitlines()]
 latest['last_loss']=next((e['loss'] for e in reversed(events) if 'loss' in e),None)
selection=Path('provenance/final-selection.json')
print(json.dumps({'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
 'remaining_minutes':(1789926479-time.time())/60,
 'estimated_compute_usd':(time.time()-1789912578.584)*.99/3600,
 'selected':json.loads(selection.read_text())['candidate'] if selection.exists() else None,
 'pipeline_exit':Path('runtime/pipeline-024.exit').read_text().strip() if Path('runtime/pipeline-024.exit').exists() else None,
 'phases':rows,'latest':latest,
 'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,temperature.gpu','--format=csv,noheader'],text=True).strip()}))
PY'''


def main():
    client=remote.connect()
    try:result=json.loads(remote.execute(client,COMMAND,timeout=30))
    finally:client.close()
    with Path(__file__).with_name('final-status.jsonl').open('a',encoding='utf-8') as output:
        output.write(json.dumps(result)+'\n')
    display={k:v for k,v in result.items() if k!='phases'}
    display['phases']={}
    for phase,row in result['phases'].items():
        last=row['last'] or {}
        display['phases'][phase]={
            'completed':row['completed'],'qualified':row['qualified'],
            'median_seconds':row['median_seconds'],'last_attempt':last.get('attempt'),
            'last_ms':last.get('timing',{}).get('candidate_graph',{}).get('median_host_ms'),
            'last_loss':last.get('loss',{}).get('candidate_graph')}
    print(json.dumps(display,indent=2))


if __name__=='__main__':
    main()
