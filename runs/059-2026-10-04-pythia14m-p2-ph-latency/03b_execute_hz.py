"""Sequential, bounded, fresh-process Ph/P2 paired benchmark."""
import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
import time
from io_utils import write

HERE=Path(__file__).resolve().parent


def jobs(smoke=False):
    if smoke:return [('0',1),('0p1',1)]
    rows=[(pair,replicate) for pair in ['0','0p1'] for replicate in range(1,4)]
    random.Random(2504).shuffle(rows)
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['smoke','scientific'],required=True)
    parser.add_argument('--deadline-epoch',type=float,required=True)
    parser.add_argument('--tag',default='001')
    args=parser.parse_args()
    if not args.tag.isalnum():parser.error('Simple unique tag required')
    output=HERE/'artifacts'/args.phase;output.mkdir(parents=True,exist_ok=True)
    lock=HERE/'artifacts/controller.lock'
    with lock.open('x') as handle:handle.write(str(__import__('os').getpid()))
    try:
        summary=[]
        for pair,replicate in jobs(args.phase=='smoke'):
            if time.time()+660>args.deadline_epoch:raise RuntimeError('Ten-minute leaf plus finalization exceeds deadline')
            attempt=f'{args.phase}-k{pair}-r{replicate}-{args.tag}'
            if (HERE/'artifacts/attempts'/attempt).exists():raise RuntimeError('Attempt already exists; retain and use a new explicit tag')
            command=[sys.executable,'-u',str(HERE/'02b_benchmark_hz.py'),'--pair',pair,
                '--replicate',str(replicate),'--attempt',attempt]
            if args.phase=='smoke':command.append('--smoke')
            started=time.monotonic()
            with (output/f'{attempt}.log').open('w',encoding='utf-8') as log:
                child=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=600)
            path=HERE/'artifacts/attempts'/attempt/'result.json'
            result=json.loads(path.read_text()) if path.exists() else {}
            row={'attempt':attempt,'returncode':child.returncode,'elapsed_seconds':time.monotonic()-started,
                'status':result.get('status'),'qualified':result.get('qualified'),'loss':result.get('loss')}
            summary.append(row);write(output/f'summary-{args.tag}.json',summary);print(json.dumps(row),flush=True)
            if child.returncode or result.get('status')!='complete':raise RuntimeError('Process failed; retain artifacts')
            if not all(q['pass']['native_graph'] for q in result['quality'].values()):raise RuntimeError('Native graph eager anchor failed')
            if args.phase=='smoke' and not result['qualified']:raise RuntimeError('Smoke failed qualification')
    finally:lock.unlink()


if __name__=='__main__':main()
