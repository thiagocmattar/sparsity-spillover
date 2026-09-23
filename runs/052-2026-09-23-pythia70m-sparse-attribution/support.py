"""Run052 paths and durable records; prior runs are read-only dependencies."""
import hashlib
import json
import os
from pathlib import Path
import time

RUN = Path(__file__).resolve().parent
BASE = Path(os.environ.get('RUN052_BASE', str(RUN.parent / '049-2026-09-22-pythia70m-short-row-limits')))

def fs(path):
    path=Path(path).absolute()
    return Path('\\\\?\\'+str(path)) if os.name=='nt' and not str(path).startswith('\\\\?\\') else path

def read(path):
    return json.loads(fs(path).read_text(encoding='utf-8'))

def write(path, value):
    path = fs(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)

def sha(path):
    digest = hashlib.sha256()
    with fs(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024), b''): digest.update(chunk)
    return digest.hexdigest()

def event(dest, stage, **fields):
    record = {'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'stage':stage, **fields}
    write(Path(dest)/'status.json',record)
    with (Path(dest)/'events.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(record,allow_nan=False)+'\n')
    print(json.dumps(record,allow_nan=False),flush=True)


def source_hashes():
    paths=list(RUN.glob('*.py'))+list(RUN.glob('*.cu'))+list(RUN.glob('*.sh'))+[RUN/'config.json']
    paths += [p for p in (RUN/'candidate14').glob('*') if p.is_file()]
    return {p.relative_to(RUN).as_posix():sha(p) for p in sorted(paths)}
