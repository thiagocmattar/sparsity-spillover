"""A bounded local smoke launcher and byte-verified result retrieval."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time
from local_support import RUN, load, write

LINUX = '/home/researcher/sparsity-spillover/' + RUN.name
PYTHON = '/opt/sparsity-gpu/venv/bin/python'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('start', 'worker', 'status', 'fetch'))
    parser.add_argument('--tag', default='smoke-001')
    args = parser.parse_args()
    assert args.tag.replace('-', '').isalnum()
    control = RUN / 'prelaunch' / ('worker-' + args.tag + '.json')
    log = RUN / 'prelaunch' / (args.tag + '.log')
    cfg = load(RUN / 'config.json')
    if args.action == 'start':
        assert not control.exists() and not log.exists(), 'Use a new attempt tag'
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('wb') as stream:
            child = subprocess.Popen([sys.executable, str(__file__), 'worker', '--tag', args.tag],
                    stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS)
        write(RUN / 'prelaunch/local-launch.json', {'pid': child.pid, 'tag': args.tag,
              'log': log.relative_to(RUN).as_posix(), 'budget_seconds': cfg['smoke_budget_seconds'],
              'authorized_scope': 'Bounded local full-model smoke and profiling; no full search.'})
        print(json.dumps({'pid': child.pid, 'tag': args.tag}))
    elif args.action == 'worker':
        started = time.monotonic(); finished = []
        result = {'status': 'running', 'tag': args.tag, 'completed': finished}
        try:
            for cid in cfg['conditions']:
                remaining = int(cfg['smoke_budget_seconds']-(time.monotonic()-started))
                assert remaining > 0, 'Local smoke budget exhausted'
                result.update(condition=cid, elapsed_seconds=time.monotonic()-started)
                write(control, result)
                command = 'source /opt/sparsity-gpu/activate.sh; cd ' + shlex.quote(LINUX) + '; exec timeout --signal=TERM --kill-after=10s ' + str(remaining) + 's ' + shlex.join([PYTHON, '-u', '02_calibrate.py', '--phase', 'smoke', '--condition', cid, '--attempt', args.tag+'-'+cid])
                leaf = subprocess.run(['wsl.exe', '-d', 'SparsityGPU', '--exec', 'bash', '-c', command])
                if leaf.returncode:
                    raise RuntimeError(f'{cid} exited {leaf.returncode}; remaining conditions not launched')
                finished.append(cid)
            result.update(status='complete')
        except Exception as exc:
            result.update(status='failed', error=str(exc))
        finally:
            result['elapsed_seconds'] = time.monotonic()-started
            write(control, result)
    elif args.action == 'status':
        print(control.read_text() if control.exists() else 'Worker status not written yet')
        if log.exists():
            print('\n'.join(log.read_text(encoding='utf-8', errors='replace').splitlines()[-8:]))
    else:
        assert load(control)['status'] in ('complete', 'failed'), 'Do not retrieve active artifacts'
        inventory_code = '''import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);tag=sys.argv[2]
files=[p for folder in (root/'artifacts/attempts').glob(tag+'-*') for p in folder.rglob('*') if p.is_file()]
rows=[]
for p in files:
 with p.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
 rows.append({'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':digest})
print(json.dumps({'files':rows}))
'''
        response = subprocess.run(['wsl.exe', '-d', 'SparsityGPU', '--exec', PYTHON, '-', LINUX, args.tag], input=inventory_code.encode(), capture_output=True, check=True)
        inventory = json.loads(response.stdout)
        assert inventory['files']
        process = subprocess.Popen(['wsl.exe', '-d', 'SparsityGPU', '--exec', 'tar', '-cf', '-', '-C', LINUX, '--files-from=-'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        process.stdin.write(('\n'.join(row['path'] for row in inventory['files'])+'\n').encode()); process.stdin.close()
        with tarfile.open(fileobj=process.stdout, mode='r|') as archive:
            archive.extractall(RUN, filter='data')
        assert process.wait()==0
        for row in inventory['files']:
            path=RUN/row['path']
            with path.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
            assert path.stat().st_size==row['bytes'] and digest==row['sha256'],row['path']
        write(RUN/'results'/(args.tag+'-retrieval.json'), inventory)
        print(json.dumps({'verified_files':len(inventory['files']), 'bytes':sum(row['bytes'] for row in inventory['files'])}))

if __name__ == '__main__':
    main()
