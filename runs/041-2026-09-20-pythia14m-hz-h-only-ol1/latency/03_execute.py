"""Bounded, sequential fresh-process evaluation of the four approved new models."""
import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def jobs(smoke=False):
    if smoke:
        return [('c00', 1), ('c03', 1)]
    pairs = [(f'c{i:02d}', r) for i in range(4) for r in range(1, 4)]
    random.Random(2504).shuffle(pairs)
    return pairs


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['smoke','scientific'],required=True)
    p.add_argument('--deadline-epoch',type=float,required=True);p.add_argument('--tag',default='001')
    args=p.parse_args()
    assert args.tag.isalnum()
    output=HERE/'artifacts'/args.phase;output.mkdir(parents=True,exist_ok=True)
    lock=HERE/'artifacts/controller.lock'
    with lock.open('x') as handle:
        handle.write(str(__import__('os').getpid()))
    try:
        summary=[]
        for condition, replicate in jobs(args.phase=='smoke'):
            if time.time()+660 > args.deadline_epoch:
                raise RuntimeError('Reserve ten-minute leaf limit and one minute for finalization before deadline')
            attempt=f'{args.phase}-{condition}-r{replicate}-{args.tag}'
            dest=HERE/'artifacts/attempts'/attempt
            if dest.exists():
                raise RuntimeError(f'Attempt already exists; retain it and use a new explicit tag: {attempt}')
            command=[sys.executable,'-u',str(HERE/'02_benchmark.py'),'--candidate','k050',
                     '--condition',condition,'--replicate',str(replicate),'--attempt',attempt,'--final']
            if args.phase=='smoke':command.append('--smoke')
            started=time.monotonic()
            with (output/f'{attempt}.log').open('w',encoding='utf-8') as log:
                child=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=600)
            path=dest/'result.json'
            result=json.loads(path.read_text()) if path.exists() else {}
            row={'attempt':attempt,'returncode':child.returncode,'elapsed_seconds':time.monotonic()-started,
                 'status':result.get('status'),'qualified':result.get('qualified'),'loss':result.get('loss')}
            summary.append(row)
            (output/f'summary-{args.tag}.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
            print(json.dumps(row),flush=True)
            if child.returncode or result.get('status')!='complete':
                raise RuntimeError('Process failure; inspect retained artifacts before retry')
            if not result.get('qualification',{}).get('native_graph',False):
                raise RuntimeError('Native reference failed its eager correctness anchor')
            if args.phase=='smoke' and not result.get('qualified'):
                raise RuntimeError('Smoke qualification failed')
    finally:
        lock.unlink()


if __name__=='__main__':main()
