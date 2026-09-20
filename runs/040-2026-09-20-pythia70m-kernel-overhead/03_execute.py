"""Recorded sequential diagnostic matrix with a reserved retrieval window."""
import argparse
import os
import random
import subprocess
import sys
import time
from io_utils import RUN, read, write


def jobs(smoke=False, optimized=False):
    cfg = read(RUN / 'config.json')
    modes = [read(RUN / 'provenance/final-selection.json')['candidate']] if optimized else cfg['diagnostic_modes']
    rows = [(condition, mode, rep) for condition in cfg['conditions']
            for mode in modes
            for rep in ([1] if smoke else range(1, cfg['process_replicates'] + 1))]
    random.Random(cfg['timing_seed']).shuffle(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['smoke', 'scientific', 'optimized'], required=True)
    parser.add_argument('--deadline-epoch', type=float, required=True)
    parser.add_argument('--tag', default='001')
    args = parser.parse_args()
    assert args.tag.isalnum()
    smoke = args.phase == 'smoke'
    if not smoke:
        summary = read(RUN / f'artifacts/smoke/summary-{args.tag}.json')
        assert len(summary) == len(jobs(True)) and all(r['qualified'] for r in summary)
        assert read(RUN / 'artifacts/operator-checks.json')['status'] == 'passed'
        assert read(RUN / 'artifacts/control-checks.json')['status'] == 'passed'
    folder = RUN / 'artifacts' / args.phase
    folder.mkdir(parents=True, exist_ok=True)
    lock = RUN / 'artifacts/controller.lock'
    with lock.open('x') as stream: stream.write(str(os.getpid()))
    if args.phase == 'optimized':
        summary = read(RUN / f'artifacts/scientific/summary-{args.tag}.json')
        assert len(summary) == len(jobs()) and all(r['qualified'] for r in summary)
    schedule = jobs(smoke, args.phase == 'optimized')
    write(folder / f'order-{args.tag}.json', schedule)
    try:
        summary = []
        for condition, mode, rep in schedule:
            timeout = 1200 if smoke else 900
            if time.time() + timeout + 1200 > args.deadline_epoch:
                raise RuntimeError('Reserved retrieval window reached; preserve partial results')
            attempt = f'{args.phase}-{condition}-{mode}-r{rep}-{args.tag}'
            if (RUN / 'artifacts/attempts' / attempt).exists():
                raise FileExistsError('Keep existing attempts; choose a new tag')
            command = [sys.executable, '-u', str(RUN / '02_benchmark.py'), '--candidate', mode,
                       '--condition', condition, '--replicate', str(rep), '--attempt', attempt]
            command.append('--smoke' if smoke else '--final')
            started = time.monotonic()
            with (folder / f'{attempt}.log').open('w', encoding='utf-8') as log:
                child = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            path = RUN / 'artifacts/attempts' / attempt / 'result.json'
            result = read(path) if path.exists() else {}
            item = {'attempt': attempt, 'condition': condition, 'mode': mode, 'replicate': rep,
                    'returncode': child.returncode, 'status': result.get('status'),
                    'qualified': result.get('qualified', False), 'loss': result.get('loss'),
                    'elapsed_seconds': time.monotonic() - started}
            summary.append(item)
            write(folder / f'summary-{args.tag}.json', summary)
            print(item, flush=True)
            if child.returncode or not item['qualified'] or item['status'] != 'complete':
                raise RuntimeError('Qualification failed; retain failure before investigating')
    finally:
        lock.unlink()


if __name__ == '__main__': main()
