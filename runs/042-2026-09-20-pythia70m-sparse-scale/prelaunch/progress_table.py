"""Compact read-only native-base comparison; keep screening and final inputs separate."""
import json
import remote

COMMAND=r'''cd /workspace/run042 && python3 - <<'PY'
import json,time,statistics
from pathlib import Path
from collections import defaultdict
groups=defaultdict(list);failures=[]
for p in Path('artifacts/attempts').glob('*/result.json'):
 d=json.loads(p.read_text());a=d['arguments']
 if a.get('smoke'):continue
 if not d.get('qualified') or d['status']!='complete':
  failures.append({'attempt':p.parent.name,'error':d.get('error')});continue
 groups[(d['condition'],d['candidate'],bool(a.get('development')))].append((p,d))
def med(rows,mode):return statistics.median(d['timing'][mode]['median_host_ms'] for _,d in rows)
baseline=[]
for condition in ('c00','c21','c16','m14-c01','m14-c20','m14-c35'):
 rows=groups.get((condition,'full',False),[])
 if rows:baseline.append({'condition':condition,'processes':len(rows),'candidate_ms':med(rows,'candidate_graph'),'native_ms':med(rows,'native_graph')})
screen=[]
for condition,k,development in groups:
 if condition!='c21' or not development:continue
 sparse=groups[(condition,k,True)];base=groups.get(('c00',k,True),[])
 latency=med(sparse,'candidate_graph')
 screen.append({'candidate':k,'latency_ms':latency,'both_checkpoints_pass':bool(base),
  'native_base_ms':med(base,'native_graph') if base else None,
  'native_base_speedup':med(base,'native_graph')/latency if base else None,
  'latest_epoch':max(p.stat().st_mtime for p,_ in sparse),
  'loss':sparse[-1][1]['loss']['candidate_graph'],'loss_delta':sparse[-1][1]['loss_delta']['candidate_graph']})
screen.sort(key=lambda x:x['latency_ms'])
operators=[]
for p in sorted(Path('artifacts/development').glob('operator-opt*.json')):
 d=json.loads(p.read_text());operators.append({'candidate':p.stem,'status':d['status']})
print(json.dumps({'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'baseline_full_validation':baseline,
 'development_ranking':screen,'operators':operators,'failures':failures,
 'pipeline_exits':{p.name:p.read_text().strip() for p in Path('runtime').glob('pipeline-*.exit')}}))
PY'''

if __name__=='__main__':
    client=remote.connect()
    data=json.loads(remote.execute(client,COMMAND))
    data['development_latest']=sorted(data['development_ranking'],key=lambda r:r['latest_epoch'],reverse=True)[:4]
    data['development_ranking']=data['development_ranking'][:6]
    data['operator_passed']=sum(r['status']=='passed' for r in data['operators'])
    data['operator_other']=[r for r in data.pop('operators') if r['status']!='passed']
    print(json.dumps(data,indent=2))
    client.close()
