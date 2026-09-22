"""Sequential bounded execution; freezes a training-only choice before final checks."""
import argparse
import os
import random
import subprocess
import sys
import time
from io_utils import RUN, read, write, record


def choose(development, cfg):
    eligible = []
    for mode in ('native_hz', 'limit16', 'limit32', 'limit64'):
        key = mode + '_graph'
        if all(r['qualification'][key] for r in development):
            target = next(r for r in development if r['arguments']['condition'] == cfg['primary_condition'])
            eligible.append((target['timing'][key]['geomean_host_ms'], mode))
    if not eligible:
        raise RuntimeError('No proposed policy qualified on all development checkpoints')
    return min(eligible)[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deadline-epoch', required=True, type=int)
    a = parser.parse_args()
    cfg = read(RUN / 'config.json')
    artifacts = RUN / 'artifacts'
    artifacts.mkdir(exist_ok=True)
    lock = artifacts / 'controller.lock'
    with lock.open('x') as f:
        f.write(str(os.getpid()))
    outcomes = []
    started = time.monotonic()

    def run(script, arguments, label, timeout):
        left = a.deadline_epoch - time.time() - cfg['recovery_reserve_seconds']
        if left < timeout:
            raise TimeoutError('Insufficient remaining budget for bounded leaf plus recovery')
        write(artifacts / 'controller-status.json', {'status': 'running', 'active': label,
              'completed_leaves': len(outcomes), 'elapsed_seconds': time.monotonic() - started})
        logfile = artifacts / (label + '.log')
        with logfile.open('x') as f:
            done = subprocess.run([sys.executable, '-u', str(RUN / script), *arguments],
                                  cwd=RUN, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
        outcomes.append({'label': label, 'returncode': done.returncode, 'log': record(logfile)})
        write(artifacts / 'controller-outcomes.json', outcomes)
        if done.returncode:
            raise RuntimeError(f'Leaf failed: {label}; see retained log')

    try:
        run('10_operators.py', [], 'operators', 1800)
        run('02_benchmark.py', ['--condition', 'c25', '--replicate', '1', '--phase', 'smoke',
                               '--attempt', 'smoke-c25-r1-001'], 'smoke', cfg['leaf_timeout_seconds'])
        smoke = read(artifacts / 'attempts/smoke-c25-r1-001/result.json')
        if not smoke['all_qualified']:
            raise RuntimeError('Preflight numerical failure; no development/final launch')
        development = []
        order = list(cfg['conditions'])
        random.Random(4901).shuffle(order)
        for cid in order:
            label = f'development-{cid}-r1-001'
            run('02_benchmark.py', ['--condition', cid, '--replicate', '1', '--phase', 'development',
                                   '--attempt', label], label, cfg['leaf_timeout_seconds'])
            development.append(read(artifacts / 'attempts' / label / 'result.json'))
        selection = {'selected': choose(development, cfg), 'criterion': cfg['selection'],
                     'evidence': [record(artifacts / 'attempts' / r['arguments']['attempt'] / 'result.json') for r in development],
                     'source_freeze': record(RUN / 'provenance/source-freeze.json'),
                     'frozen_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        if (RUN / 'provenance/selection.json').exists():
            raise FileExistsError('Selection is immutable')
        write(RUN / 'provenance/selection.json', selection)
        jobs = [(cid, rep) for cid in cfg['conditions'] for rep in range(1, 4)]
        random.Random(4902).shuffle(jobs)
        write(artifacts / 'final-order.json', jobs)
        for cid, rep in jobs:
            label = f'final-{cid}-r{rep}-001'
            run('02_benchmark.py', ['--condition', cid, '--replicate', str(rep), '--phase', 'final',
                                   '--attempt', label], label, cfg['leaf_timeout_seconds'])
        write(artifacts / 'controller-status.json', {'status': 'complete', 'completed_leaves': len(outcomes),
              'elapsed_seconds': time.monotonic() - started})
    except Exception as exc:
        write(artifacts / 'controller-status.json', {'status': 'failed', 'error': str(exc),
              'completed_leaves': len(outcomes), 'elapsed_seconds': time.monotonic() - started})
        raise
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
