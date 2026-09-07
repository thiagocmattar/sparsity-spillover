"""Read-only bounded status for the single final matrix."""
import json
from pathlib import Path
import time

RUN=Path(__file__).resolve().parent
def read_json(path):
    with path.open(encoding='utf-8') as stream:return json.load(stream)

path=RUN/'artifacts/final-matrix-002/status.json'
status=read_json(path)
status['status_age_seconds']=time.time()-path.stat().st_mtime
outcomes=status.pop('outcomes',[])
status['complete_outcomes']=len(outcomes)
if outcomes and 'current' in status:
    replicate=status['current']['replicate']
    comparable=[r['elapsed_seconds'] for r in outcomes if (r['replicate']>=2 if replicate>=2 else r['replicate']==1)]
    if comparable:
        status['phase_adjusted_remaining_seconds']=max(0,(status['total']-len(outcomes))*sum(comparable)/len(comparable)-status['status_age_seconds'])
status['infrastructure_failures']=[r['attempt'] for r in outcomes if r['status']!='complete']
status['new_kernel_quality_failures']=[r['attempt'] for r in outcomes if r.get('qualified',{}).get('sparse_graph') is False]
if outcomes:status['last_loss']=outcomes[-1].get('loss',{}).get('native')
if 'current' in status:
    leaf=RUN/'artifacts'/status['current']['attempt']/'status.json'
    if leaf.exists():
        row=read_json(leaf)
        status['leaf']={k:v for k,v in row.items() if k not in {'timing','loss','qualified'}}
        if 'loss' in row:status['leaf']['native_loss']=row['loss']['native']
print(json.dumps(status))
