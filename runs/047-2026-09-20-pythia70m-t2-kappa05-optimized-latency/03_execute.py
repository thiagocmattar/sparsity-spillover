"""Sequential matched comparisons with unique durable attempts and a stop reserve."""
import argparse
import os
import random
import subprocess
import sys
import time
from io_utils import RUN, read, write, verify


def jobs(config, phase):
    conditions = config['conditions']
    replicates = (1,) if phase == 'preflight' else range(1, config['process_replicates']+1)
    rows = [(condition, replicate) for condition in conditions for replicate in replicates]
    random.Random(config['runtime_seed']).shuffle(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('preflight', 'final'), required=True)
    parser.add_argument('--tag', default='001')
    parser.add_argument('--deadline-epoch', type=float, required=True)
    args = parser.parse_args()
    if not args.tag.isalnum():
        parser.error('Alphanumeric unique tag required')
    config = read(RUN/'config.json')
    for row in read(RUN/'provenance/source-freeze.json')['files']:
        verify(row)
    out = RUN/'artifacts'/args.phase
    out.mkdir(parents=True, exist_ok=True)
    summary = out/f'summary-{args.tag}.json'
    if summary.exists():
        raise FileExistsError('Use a fresh attempt tag')
    (RUN/'runtime').mkdir(exist_ok=True)
    order = jobs(config, args.phase)
    write(out/f'order-{args.tag}.json', order)
    lock = RUN/'artifacts/controller.lock'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    results = []
    try:
        for condition, replicate in order:
            remaining = args.deadline_epoch-time.time()-config['recovery_reserve_seconds']
            if remaining < config['leaf_timeout_seconds']:
                raise TimeoutError('Insufficient time for bounded leaf and recovery reserve')
            attempt = f'{args.phase}-{condition}-r{replicate}-{args.tag}'
            command = [sys.executable, '-u', str(RUN/'02_benchmark.py'), '--condition', condition,
                       '--replicate', str(replicate), '--attempt', attempt,
                       '--smoke' if args.phase == 'preflight' else '--final']
            started = time.monotonic()
            with (RUN/'runtime'/f'{attempt}.log').open('x') as log:
                completed = subprocess.run(command, cwd=RUN, stdout=log, stderr=subprocess.STDOUT,
                                           timeout=config['leaf_timeout_seconds'])
            result = read(RUN/'artifacts/attempts'/attempt/'result.json')
            seconds = time.monotonic()-started
            results.append({'attempt': attempt, 'condition': condition, 'replicate': replicate,
                            'returncode': completed.returncode, 'seconds': seconds,
                            'qualified': result.get('qualified', False), 'status': result['status'],
                            'loss': result.get('loss'), 'timing': result.get('timing'),
                            'error': result.get('error')})
            write(summary, results)
            write(RUN/'artifacts/progress.json', {
                'phase': args.phase, 'completed': len(results), 'total': len(order),
                'latest': results[-1], 'evaluation_input_tokens_per_second':
                    result.get('validation_blocks', 0)*2048/seconds,
                'remaining_seconds': (len(order)-len(results))*sum(r['seconds'] for r in results)/len(results)})
            print(results[-1], flush=True)
            # Numerical failures remain evidence; collect the rest without retuning.
            if result['status'] != 'complete' or (args.phase == 'preflight' and completed.returncode):
                raise RuntimeError('Infrastructure/preflight failure requires investigation')
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
