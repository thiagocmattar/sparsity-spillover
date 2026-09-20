"""Freeze the lowest T7/Ph development geometric-mean latency, after both checks."""
from pathlib import Path
import math
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from io_utils import RUN,read,write,record

assert not (RUN/'artifacts/controller.lock').exists()
assert read(RUN/'artifacts/development/summary-001.json')['rows']
ranking=[]
for identifier in ('opt001','opt002','opt003'):
    row={'candidate':identifier,'conditions':{}}
    if identifier!='opt001':assert read(RUN/f'artifacts/development/operator-{identifier}.json')['status']=='passed'
    for condition in ('c00','c21'):
        folder=RUN/f'artifacts/attempts/development-{condition}-{identifier}-r1-001'
        result=read(folder/'result.json');timing=read(folder/'timing.json')
        assert result['qualified'] and result['status']=='complete' and result['arguments']['development']
        assert result['evaluation_split']=='training-development' and result['validation_blocks']==16
        assert timing['indices']==list(range(16))
        means={}
        for mode in ('candidate_graph','native_graph','frozen_graph'):
            samples=[x['host_ms'] for x in timing['samples'] if x['mode']==mode]
            assert len(samples)==80 and all(math.isfinite(x) and x>0 for x in samples)
            means[mode]=math.exp(math.fsum(map(math.log,samples))/len(samples))
        row['conditions'][condition]={'geometric_mean_host_ms':means,
             'result':record(folder/'result.json'),'timing':record(folder/'timing.json')}
    ranking.append(row)
ranking.sort(key=lambda x:x['conditions']['c21']['geometric_mean_host_ms']['candidate_graph'])
assert ranking[0]['candidate']=='opt002'
write(RUN/'provenance/development-ranking.json',{'criterion':'Lowest qualified T7/Ph training-development latency; no final evaluation used.',
      'ranking':ranking,'selected':ranking[0]['candidate'],'script':record(__file__)})
subprocess.run([sys.executable,str(RUN/'11_candidate.py'),'freeze',ranking[0]['candidate']],check=True)
print({x['candidate']:x['conditions']['c21']['geometric_mean_host_ms']['candidate_graph'] for x in ranking},flush=True)
