"""Six fixed fresh processes; unchanged cohort, no concurrent GPU timing."""
import argparse
import json
import os
import random
import subprocess
import sys
import time
from io_utils import RUN, read, write


def jobs(smoke=False):
    if smoke:
        return [('0',1),('0p1',1)]
    rows=[(pair,r) for pair in ['0','0p1'] for r in range(1,4)]
    random.Random(2504).shuffle(rows)
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['smoke','scientific'],required=True)
    parser.add_argument('--deadline-epoch',type=float,required=True)
    parser.add_argument('--tag',default='001')
    args=parser.parse_args()
    if not args.tag.isalnum():
        parser.error('Simple unique tag required')
    output=RUN/'artifacts'/args.phase;output.mkdir(parents=True,exist_ok=True)
    lock=RUN/'artifacts/controller.lock'
    with lock.open('x') as handle:
        handle.write(str(os.getpid()))
    summary=[]
    timeout=read(RUN/'config.json')['leaf_timeout_seconds']
    try:
        for pair,r in jobs(args.phase=='smoke'):
            if time.time()+timeout+120>args.deadline_epoch:
                raise RuntimeError('Leaf reservation exceeds lease; leave time to retrieve artifacts')
            attempt=f'{args.phase}-k{pair}-r{r}-{args.tag}'
            command=[sys.executable,'-u',str(RUN/'02_benchmark.py'),'--pair',pair,
                     '--replicate',str(r),'--attempt',attempt]
            if args.phase=='smoke':
                command.append('--smoke')
            started=time.monotonic()
            with (output/f'{attempt}.log').open('x',encoding='utf-8') as log:
                child=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
            result=read(RUN/'artifacts/attempts'/attempt/'result.json')
            row={'attempt':attempt,'returncode':child.returncode,'elapsed_seconds':time.monotonic()-started,
                 'status':result['status'],'qualified':result.get('qualified'),'loss':result.get('loss')}
            summary.append(row);write(output/f'summary-{args.tag}.json',summary)
            print(json.dumps(row),flush=True)
            if child.returncode or not result.get('qualified') or result['status']!='complete':
                raise RuntimeError('Qualification/process failed; retain this attempt and stop')
    finally:
        lock.unlink()


if __name__=='__main__':
    main()
