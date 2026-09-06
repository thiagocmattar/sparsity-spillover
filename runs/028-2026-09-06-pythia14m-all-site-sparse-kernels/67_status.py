"""Read-only bounded status for the single final matrix."""
import json
from common import RUN,read_json

path=RUN/'artifacts/final-matrix-001/status.json'
status=read_json(path)
outcomes=status.pop('outcomes',[])
status['complete_outcomes']=len(outcomes)
status['infrastructure_failures']=[r['attempt'] for r in outcomes if r['status']!='complete']
status['new_kernel_quality_failures']=[r['attempt'] for r in outcomes if r.get('qualified',{}).get('sparse_graph') is False]
if outcomes:status['last_loss']=outcomes[-1].get('loss',{}).get('native')
if 'current' in status:
    leaf=RUN/'artifacts'/status['current']['attempt']/'status.json'
    if leaf.exists():status['leaf']=read_json(leaf)
print(json.dumps(status))
