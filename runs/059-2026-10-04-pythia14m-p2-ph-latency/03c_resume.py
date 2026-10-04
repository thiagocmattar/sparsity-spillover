"""Resume only unfinished fixed jobs, using a shorter bounded cached-process lease."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from io_utils import RUN, read, write, record, module


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--deadline-epoch',type=float,required=True)
    parser.add_argument('--leaf-seconds',type=int,required=True)
    args=parser.parse_args()
    if not 120<=args.leaf_seconds<=600:parser.error('Bounded cached-process limit required')
    smoke=read(RUN/'artifacts/smoke/summary-004.json')
    if len(smoke)!=2 or not all(row['returncode']==0 and row['qualified'] for row in smoke):
        raise RuntimeError('Both original smoke checks must pass first')
    controller=module('run059_original_controller',RUN/'03b_execute_hz.py')
    previous=read(RUN/'artifacts/scientific/summary-004.json')
    completed=set()
    for row in previous:
        result=read(RUN/'artifacts/attempts'/row['attempt']/'result.json')
        if row['returncode'] or result['status']!='complete':raise RuntimeError('Failed science requires an explicit new retry')
        if not all(q['pass']['native_graph'] for q in result['quality'].values()):
            raise RuntimeError('Original native graph eager anchor failed')
        if result['config']!=record(RUN/'config.json'):raise RuntimeError('Scientific config changed')
        completed.add((result['arguments']['pair'],result['arguments']['replicate']))
    if len(completed)!=len(previous):raise RuntimeError('Duplicate original process')
    pending=[job for job in controller.jobs() if job not in completed]
    lock=RUN/'artifacts/controller.lock'
    with lock.open('x') as handle:handle.write(str(__import__('os').getpid()))
    output=RUN/'artifacts/scientific';new=[]
    try:
        write(output/'resume-005.json',{'previous':previous,'pending':pending,
            'leaf_seconds':args.leaf_seconds,'deadline_epoch':args.deadline_epoch,
            'scientific_protocol_changed':False})
        for pair,replicate in pending:
            if time.time()+args.leaf_seconds+60>args.deadline_epoch:
                raise RuntimeError('Cached leaf plus retrieval reserve exceeds deadline')
            attempt=f'scientific-k{pair}-r{replicate}-005'
            if (RUN/'artifacts/attempts'/attempt).exists():raise RuntimeError('Preserve existing attempt')
            command=[sys.executable,'-u',str(RUN/'02b_benchmark_hz.py'),'--pair',pair,
                '--replicate',str(replicate),'--attempt',attempt]
            started=time.monotonic()
            with (output/f'{attempt}.log').open('w',encoding='utf-8') as log:
                child=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=args.leaf_seconds)
            result=read(RUN/'artifacts/attempts'/attempt/'result.json')
            row={'attempt':attempt,'returncode':child.returncode,'elapsed_seconds':time.monotonic()-started,
                'status':result['status'],'qualified':result.get('qualified'),'loss':result.get('loss')}
            new.append(row);write(output/'summary-005.json',new)
            write(output/'cohort-summary.json',previous+new);print(json.dumps(row),flush=True)
            if child.returncode or result['status']!='complete':raise RuntimeError('Retain failed process')
            if not all(q['pass']['native_graph'] for q in result['quality'].values()):
                raise RuntimeError('Native graph eager anchor failed')
    finally:lock.unlink()


if __name__=='__main__':main()
