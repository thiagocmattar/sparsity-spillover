"""Sequential smoke and scientific evaluations, bounded by a retrieval deadline."""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from controls import MODES

HERE = Path(__file__).resolve().parent


def jobs(smoke=False):
    if smoke:
        return [(mode, 1) for mode in MODES]
    pairs = [(mode, rep) for rep in range(1, 4) for mode in MODES]
    random.Random(2504).shuffle(pairs)
    return pairs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['smoke', 'scientific'], required=True)
    p.add_argument('--deadline-epoch', type=float, required=True)
    p.add_argument('--tag', default='001')
    args = p.parse_args()
    if not args.tag.isalnum(): p.error('Alphanumeric tag required')
    smoke = args.phase == 'smoke'
    if not smoke:
        summary = json.loads((HERE / f'artifacts/smoke/summary-{args.tag}.json').read_text())
        assert len(summary) == len(MODES) and all(r['qualified'] for r in summary)
        assert json.loads((HERE / 'artifacts/cuda-controls.json').read_text())['status'] == 'passed'
    output = HERE / 'artifacts' / args.phase
    output.mkdir(parents=True, exist_ok=True)
    lock = HERE / 'artifacts/controller.lock'
    with lock.open('x') as f: f.write(str(os.getpid()))
    try:
        summary = []
        for mode, replicate in jobs(smoke):
            timeout = 1200 if smoke else 600
            if time.time() + timeout + 60 > args.deadline_epoch:
                raise RuntimeError('Insufficient time for next process before retrieval deadline')
            attempt = f'{args.phase}-{mode}-r{replicate}-{args.tag}'
            dest = HERE / 'artifacts/attempts' / attempt
            if dest.exists(): raise FileExistsError(f'Retain prior attempt; use a new tag: {attempt}')
            command = [sys.executable, '-u', str(HERE / '02_benchmark.py'), '--candidate', mode,
                       '--condition', 'c30', '--replicate', str(replicate), '--attempt', attempt, '--final']
            if smoke: command.append('--smoke')
            started = time.monotonic()
            with (output / f'{attempt}.log').open('w', encoding='utf-8') as log:
                child = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            result_path = dest / 'result.json'
            result = json.loads(result_path.read_text()) if result_path.exists() else {}
            row = {'attempt': attempt, 'mode': mode, 'replicate': replicate,
                   'returncode': child.returncode, 'elapsed_seconds': time.monotonic() - started,
                   'status': result.get('status'), 'qualified': result.get('qualified', False),
                   'loss': result.get('loss')}
            summary.append(row)
            (output / f'summary-{args.tag}.json').write_text(json.dumps(summary, indent=2) + '\n')
            print(json.dumps(row), flush=True)
            if child.returncode or not row['qualified'] or row['status'] != 'complete':
                raise RuntimeError('Qualification/process failure; preserve evidence and investigate')
    finally:
        lock.unlink()


if __name__ == '__main__': main()
