"""One sequential controller; every setting gets three independent processes."""
import argparse
import json
import random
import subprocess
import sys
import time
from io_utils import RUN, read, write


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['preflight','scientific'],required=True)
    parser.add_argument('--deadline-epoch',type=float,required=True)
    parser.add_argument('--tag',default='001')
    args=parser.parse_args()
    manifest=read(RUN/'provenance/inputs.json')
    if args.phase=='preflight':
        # Both architectures/activations, p=0,.5,.9; 8 blocks each.
        jobs=[(c,1) for c in manifest['conditions'] if c['p'] in (0,.5,.9)]
    else:
        jobs=[(c,r) for c in manifest['conditions'] for r in range(1,4)]
        random.Random(2504).shuffle(jobs)
    out=RUN/'artifacts'/args.phase;out.mkdir(parents=True,exist_ok=True)
    lock=RUN/'artifacts/controller.lock'
    with lock.open('x') as f:f.write(str(__import__('os').getpid()))
    summary=[]
    try:
        for condition,replicate in jobs:
            if time.time()+2100>args.deadline_epoch:
                raise RuntimeError('Reserve 30min leaf plus 5min transfer before deadline')
            attempt=f'{args.phase}-{condition["id"]}-r{replicate}-{args.tag}'
            command=[sys.executable,'-u',str(RUN/'02_benchmark.py'),'--condition',condition['id'],
                '--candidate',condition['candidate'],'--replicate',str(replicate),'--attempt',attempt,'--final']
            if args.phase=='preflight':command.append('--smoke')
            started=time.monotonic()
            with (out/f'{attempt}.log').open('w',encoding='utf-8') as log:
                child=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
            result=read(RUN/'artifacts/attempts'/attempt/'result.json')
            row={'attempt':attempt,'condition':condition['id'],'status':result['status'],
                'qualified':result.get('qualified'),'loss':result.get('loss'),
                'seconds':time.monotonic()-started,'returncode':child.returncode}
            summary.append(row);write(out/f'summary-{args.tag}.json',summary)
            print(json.dumps(row),flush=True)
            if child.returncode or result['status']!='complete':
                raise RuntimeError('Process failed; preserve attempt and diagnose')
            if not result['qualification']['native_graph']:
                raise RuntimeError('Native graph failed the eager anchor')
            if args.phase=='preflight' and not result['qualified']:
                raise RuntimeError('Preflight qualification failed')
        write(out/f'complete-{args.tag}.json',{'status':'complete','processes':len(summary)})
    finally:
        lock.unlink()


if __name__=='__main__':main()
