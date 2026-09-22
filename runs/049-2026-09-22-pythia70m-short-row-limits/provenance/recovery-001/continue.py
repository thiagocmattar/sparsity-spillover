"""Resume the unchanged final schedule after an import-only infrastructure failure."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
RUN = HERE.parents[1]
sys.path.insert(0, str(RUN))
from io_utils import read, write, record

p = argparse.ArgumentParser(); p.add_argument('--deadline-epoch', required=True, type=int); a = p.parse_args()
cfg = read(RUN/'config.json')
dest = RUN/'artifacts/recovery-001'; dest.mkdir(parents=True, exist_ok=True)
lock = RUN/'artifacts/controller.lock'
with lock.open('x') as f: f.write(str(os.getpid()))
outcomes = []
started = time.monotonic()

def execute(script, args, label):
    if a.deadline_epoch-time.time() < cfg['leaf_timeout_seconds']+cfg['recovery_reserve_seconds']:
        raise TimeoutError('Insufficient remaining leaf and retrieval budget')
    write(dest/'controller-status.json', {'status':'running','active':'recovery-001/'+label,'completed_leaves':len(outcomes),'elapsed_seconds':time.monotonic()-started})
    path = dest/(label+'.log')
    with path.open('x') as output:
        result = subprocess.run([sys.executable,'-u',str(script),*args],cwd=RUN,stdout=output,stderr=subprocess.STDOUT,timeout=cfg['leaf_timeout_seconds'])
    outcomes.append({'label':label,'returncode':result.returncode,'log':record(path)})
    write(dest/'controller-outcomes.json',outcomes)
    if result.returncode: raise RuntimeError('Recovery leaf failed: '+label)

try:
    original = read(RUN/'artifacts/attempts/final-c24-r1-001/result.json')
    assert original['status']=='failed' and original['error']=="collect() got an unexpected keyword argument 'blocks'"
    execute(HERE/'diagnostic_smoke.py', [], 'diagnostic-smoke')
    jobs = read(RUN/'artifacts/final-order.json')
    assert jobs[0]==['c00',3]
    for cid, rep in jobs[1:]:
        attempt = f'final-{cid}-r{rep}-'+('002' if [cid,rep]==['c24',1] else '001')
        execute(HERE/'benchmark.py',['--condition',cid,'--replicate',str(rep),'--phase','final','--attempt',attempt],attempt)
    write(dest/'controller-status.json',{'status':'complete','completed_leaves':len(outcomes),'elapsed_seconds':time.monotonic()-started})
except Exception as exc:
    write(dest/'controller-status.json',{'status':'failed','error':str(exc),'completed_leaves':len(outcomes),'elapsed_seconds':time.monotonic()-started})
    raise
finally: lock.unlink()
