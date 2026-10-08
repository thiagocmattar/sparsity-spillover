"""Detached bounded sequential execution; durable attempt logs."""
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT=Path(__file__).parent
deadline=float(os.environ['RUN065_DEADLINE'])
(ROOT/'artifacts').mkdir(exist_ok=True)
jobs=[(size,p,r,False) for size in ['31m','70m'] for p in ['0','0p1'] for r in [1,2,3]]
random.Random(2504).shuffle(jobs)
jobs=[(size,p,1,True) for size in ['31m','70m'] for p in ['0','0p1']]+jobs
summary=[]
for size,pair,replicate,smoke in jobs:
    if time.time()+1920>deadline:raise RuntimeError('Deadline leaves insufficient leaf/retrieval reserve')
    name=f'{"smoke" if smoke else "full"}-{size}-k{pair}-r{replicate}-001'
    command=[sys.executable,'-u',str(ROOT/'02_profile.py'),'--size',size,'--pair',pair,'--replicate',str(replicate)]
    if smoke:command.append('--smoke')
    with (ROOT/'artifacts'/(name+'.log')).open('x') as log:
        completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
    result=json.loads((ROOT/'artifacts/attempts'/name/'result.json').read_text())
    summary.append({'attempt':name,'returncode':completed.returncode,'status':result['status'],'qualified':result.get('qualified'),'elapsed_seconds':result['elapsed_seconds']})
    (ROOT/'artifacts/controller-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary[-1]),flush=True)
    if completed.returncode or result['status']!='complete':raise RuntimeError('Leaf failed; no later condition launched')
