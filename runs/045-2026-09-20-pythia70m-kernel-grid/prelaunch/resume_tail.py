"""Finish only a deadline-reserved tail, preserving the original evidence and cap.

The frozen controller reserves its entire 20-minute leaf timeout before each
start. If it stops for that reason, this infrastructure-only continuation uses
the smaller remaining allowance, still reserving the final 20 minutes for
recovery. It never changes benchmark inputs, code, coverage or tolerances.
"""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from io_utils import RUN, read, write, verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deadline-epoch', type=int, required=True)
    args = parser.parse_args()
    # The authorized deadline is unchanged from the original deployment.
    assert args.deadline_epoch == 1789941035
    control = Path('/workspace/run045-control')
    assert (control/'pipeline.exit').exists()
    assert int((control/'pipeline.exit').read_text()) != 0
    assert 'Insufficient time for bounded leaf and recovery reserve' in (control/'pipeline.log').read_text()
    config = read(RUN/'config.json')
    for row in read(RUN/'provenance/source-freeze.json')['files']:
        verify(row)
    final = RUN/'artifacts/final'
    prior = read(final/'summary-001.json')
    order = read(final/'order-001.json')
    assert len(prior) < len(order) == 78
    assert [[x['condition'], x['replicate']] for x in prior] == order[:len(prior)]
    assert all(x['status'] == 'complete' for x in prior)
    summary = final/'summary-tail001.json'
    if summary.exists():
        raise FileExistsError('Tail attempt already exists; retain it for inspection')
    lock = RUN/'artifacts/controller.lock'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    records = list(prior)
    write(final/'order-tail001.json', order)
    write(RUN/'provenance/tail-continuation.json', {
        'reason': 'Conservative controller start reserve; bounded warm tail only',
        'original_deadline_epoch': args.deadline_epoch,
        'recovery_reserve_seconds': config['recovery_reserve_seconds'],
        'retained_processes': len(prior), 'new_attempt_tag': 'tail001',
        'benchmark_sources_unchanged': True,
        'controller_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    try:
        for condition, replicate in order[len(prior):]:
            available = args.deadline_epoch-time.time()-config['recovery_reserve_seconds']
            timeout = min(config['leaf_timeout_seconds'], available-10)
            if timeout < 180:
                raise TimeoutError('Less than three minutes remain before recovery reserve')
            attempt = f'final-{condition}-r{replicate}-tail001'
            command = [sys.executable, '-u', str(RUN/'02_benchmark.py'), '--condition', condition,
                       '--replicate', str(replicate), '--attempt', attempt, '--final']
            started = time.monotonic()
            with (RUN/'runtime'/f'{attempt}.log').open('x') as log:
                completed = subprocess.run(command, cwd=RUN, stdout=log, stderr=subprocess.STDOUT,
                                           timeout=timeout)
            result = read(RUN/'artifacts/attempts'/attempt/'result.json')
            seconds = time.monotonic()-started
            records.append(dict(attempt=attempt, condition=condition, replicate=replicate,
                returncode=completed.returncode, seconds=seconds, qualified=result.get('qualified',False),
                status=result['status'], loss=result.get('loss'), timing=result.get('timing'),
                error=result.get('error'), infrastructure_timeout_seconds=timeout))
            write(summary, records)
            write(RUN/'artifacts/progress.json', dict(phase='final-tail',completed=len(records),total=78,
                latest=records[-1],evaluation_input_tokens_per_second=result.get('validation_blocks',0)*2048/seconds,
                remaining_seconds=(78-len(records))*seconds))
            print(records[-1],flush=True)
            if result['status'] != 'complete':
                raise RuntimeError('Incomplete leaf; preserve evidence and investigate')
    finally:
        lock.unlink(missing_ok=True)
    subprocess.run([sys.executable,'06_reduce.py','--tag','tail001'],cwd=RUN,check=True)
    subprocess.run([sys.executable,'08_collect.py','--tag','tail001'],cwd=RUN,check=True)


if __name__ == '__main__':
    main()
