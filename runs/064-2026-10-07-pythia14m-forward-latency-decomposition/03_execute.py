"""Detached bounded sequential execution; durable attempt logs."""
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT=Path(__file__).parent
deadline=float(os.environ['RUN064_DEADLINE'])
(ROOT/'artifacts').mkdir(exist_ok=True)
jobs=[(p,r,False) for p in ['0','0p1'] for r in [1,2,3]]
random.Random(2504).shuffle(jobs)
jobs=[('0',1,True),('0p1',1,True)]+jobs
summary=[]
for pair,replicate,smoke in jobs:
    if time.time()+1200>deadline:raise RuntimeError('Deadline leaves insufficient leaf/retrieval reserve')
    name=f'{"smoke" if smoke else "full"}-k{pair}-r{replicate}-001'
    command=[sys.executable,'-u',str(ROOT/'02_profile.py'),'--pair',pair,'--replicate',str(replicate)]
    if smoke:command.append('--smoke')
    with (ROOT/'artifacts'/(name+'.log')).open('x') as log:
        completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=1080)
    result=json.loads((ROOT/'artifacts/attempts'/name/'result.json').read_text())
    summary.append({'attempt':name,'returncode':completed.returncode,'status':result['status'],'qualified':result.get('qualified'),'elapsed_seconds':result['elapsed_seconds']})
    (ROOT/'artifacts/controller-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary[-1]),flush=True)
    if completed.returncode or result['status']!='complete':raise RuntimeError('Leaf failed; no later condition launched')
