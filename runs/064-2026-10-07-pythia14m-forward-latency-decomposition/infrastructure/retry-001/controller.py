"""Six fresh processes, same lease and inputs; failed001 stays immutable."""
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
deadline=float(os.environ['RUN064_DEADLINE'])
jobs=[(p,r) for p in ['0','0p1'] for r in [1,2,3]]
random.Random(2504).shuffle(jobs);summary=[]
for pair,replicate in jobs:
    if time.time()+1200>deadline:raise RuntimeError('Deadline reserve')
    name=f'full-k{pair}-r{replicate}-002'
    command=[sys.executable,'-u',str(Path(__file__).with_name('profile.py')),'--pair',pair,'--replicate',str(replicate),'--tag','002']
    with (ROOT/'artifacts'/(name+'.log')).open('x') as log:
        completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=1080)
    result=json.loads((ROOT/'artifacts/attempts'/name/'result.json').read_text())
    summary.append({'attempt':name,'returncode':completed.returncode,'status':result['status'],'qualified':result.get('qualified'),'elapsed_seconds':result['elapsed_seconds']})
    (ROOT/'artifacts/controller-summary-002.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary[-1]),flush=True)
    if completed.returncode or result['status']!='complete':raise RuntimeError('Leaf failed')
