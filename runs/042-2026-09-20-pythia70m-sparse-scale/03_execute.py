"""Bounded sequential jobs on one GPU; every attempt survives terminal loss."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import random
from io_utils import RUN, read, write, verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('baseline','development','final','decomposition'), required=True)
    parser.add_argument('--candidates', nargs='+', default=[f'opt{i:03d}' for i in range(1,7)])
    parser.add_argument('--tag', default='001')
    parser.add_argument('--deadline-epoch', type=float, required=True)
    args = parser.parse_args()
    cfg = read(RUN/'config.json')
    out = RUN/'artifacts'/args.phase
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    if args.phase == 'baseline':
        jobs = [(c,'full',r) for c in cfg['conditions'] for r in (1,2,3)]
    elif args.phase == 'development':
        jobs = [(c,k,1) for k in args.candidates for c in ('c00','c21')]
    elif args.phase == 'final':
        k = read(RUN/'provenance/final-selection.json')['candidate']
        jobs = [(c,k,r) for c in ('c00','c21','c16') for r in (1,2,3)]
    else:
        read(RUN/'provenance/final-selection.json')
        modes = ('selected-no-skip','selected-native-hz','selected-native-attention',
                 'selected-native-am','selected-native-norm','selected-native-rope',
                 'selected-native-head')
        jobs = [(c,k,r) for c in ('c00','c21') for k in modes for r in (1,2,3)]
        jobs += [('m14-c20',k,r) for k in ('hz-skips-off','native-hz') for r in (1,2,3)]
        jobs += [('m14-c01','native-hz',r) for r in (1,2,3)]
    random.Random(2801).shuffle(jobs)
    write(out/f'order-{args.tag}.json', jobs)
    rows = []
    lock = RUN/'artifacts/controller.lock'
    with lock.open('x') as f: f.write(str(os.getpid()))
    try:
        for condition,candidate,replicate in jobs:
            if args.phase == 'development' and (RUN/'candidates'/candidate/'spec.json').exists():
                operator = RUN/'artifacts/development'/f'operator-{candidate}.json'
                regression = RUN/'artifacts/development'/f'operator-regression-qualification-{candidate}.json'
                eligible=operator.exists() and read(operator).get('status') == 'passed'
                if not eligible and regression.exists():
                    evidence=read(regression)
                    assert evidence['status']=='passed-bitwise-regression-control'
                    assert evidence['candidate']==candidate
                    for name in ('manifest','baseline_audit','candidate_audit'):verify(evidence[name])
                    eligible=True
                if not eligible:
                    rows.append({'condition':condition,'candidate':candidate,'status':'operator-check-failed-or-missing'})
                    write(out/f'summary-{args.tag}.json',rows)
                    continue
            if time.time() > args.deadline_epoch-1200:
                raise TimeoutError('Recovery reserve reached')
            attempt = f'{args.phase}-{condition}-{candidate}-r{replicate}-{args.tag}'
            command = [sys.executable,'-u',str(RUN/'02_benchmark.py'),'--condition',condition,
                       '--candidate',candidate,'--replicate',str(replicate),'--attempt',attempt]
            if args.phase == 'development': command += ['--development']
            if args.phase == 'final': command += ['--final']
            started = time.monotonic()
            with (RUN/'runtime'/f'{attempt}.log').open('x') as log:
                completed = subprocess.run(command, cwd=RUN, stdout=log, stderr=subprocess.STDOUT,
                                           timeout=min(1200,args.deadline_epoch-time.time()-600))
            path = RUN/'artifacts/attempts'/attempt/'result.json'
            result = read(path) if path.exists() else {}
            row = {'attempt':attempt,'condition':condition,'candidate':candidate,'replicate':replicate,
                   'returncode':completed.returncode,'seconds':time.monotonic()-started,
                   'qualified':result.get('qualified',False),'loss':result.get('loss'),
                   'timing':result.get('timing'),'error':result.get('error')}
            rows.append(row)
            write(out/f'summary-{args.tag}.json', rows)
            print(row, flush=True)
            if completed.returncode and args.phase != 'development':
                raise RuntimeError('Baseline/final failure requires investigation')
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__': main()
