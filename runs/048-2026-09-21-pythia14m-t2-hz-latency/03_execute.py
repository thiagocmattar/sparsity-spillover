"""Bounded sequential fresh-process execution; all modes are paired within a process."""
import argparse
import os
import subprocess
import sys
import time
from io_utils import RUN, read, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['smoke', 'final'], required=True)
    parser.add_argument('--deadline-epoch', required=True, type=float)
    parser.add_argument('--tag', default='001')
    args = parser.parse_args()
    assert args.tag.isalnum()
    assert read(RUN/'artifacts/cuda-controls.json')['status'] == 'passed'
    if args.phase == 'final':
        smoke = read(RUN/f'artifacts/smoke/summary-{args.tag}.json')
        assert len(smoke) == 1 and smoke[0]['status'] == 'complete'
    output = RUN/'artifacts'/args.phase
    output.mkdir(parents=True, exist_ok=True)
    lock = RUN/'artifacts/controller.lock'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    rows = []
    try:
        for replicate in ([1] if args.phase == 'smoke' else [1, 2, 3]):
            timeout = 1200
            if time.time()+timeout+1200 > args.deadline_epoch:
                raise RuntimeError('Insufficient time before collection reserve')
            attempt = f'{args.phase}-r{replicate}-{args.tag}'
            command = [sys.executable, '-u', str(RUN/'02_benchmark.py'), '--replicate',
                       str(replicate), '--attempt', attempt]
            if args.phase == 'smoke':
                command.append('--smoke')
            started = time.monotonic()
            with (output/f'{attempt}.log').open('w', encoding='utf-8') as log:
                child = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            result = read(RUN/'artifacts/attempts'/attempt/'result.json')
            row = {'attempt': attempt, 'replicate': replicate, 'returncode': child.returncode,
                   'seconds': time.monotonic()-started,
                   **{k: result.get(k) for k in ['status', 'qualified', 'loss', 'error']}}
            rows.append(row)
            write(output/f'summary-{args.tag}.json', rows)
            write(RUN/'artifacts/progress.json', {'phase': args.phase, 'completed': len(rows), 'latest': row})
            print(row, flush=True)
            if child.returncode or result['status'] != 'complete':
                raise RuntimeError('Scientific failure; retain every artifact')
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
