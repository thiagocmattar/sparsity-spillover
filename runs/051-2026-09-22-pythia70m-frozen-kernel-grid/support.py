"""Run050 paths and durable records, independent of archived module names."""
import hashlib
import json
import os
from pathlib import Path
import time

RUN = Path(__file__).resolve().parent
BASE = Path(os.environ.get('RUN050_BASE', str(RUN.parent / '049-2026-09-22-pythia70m-short-row-limits')))

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024), b''): digest.update(chunk)
    return digest.hexdigest()

def event(dest, stage, **fields):
    record = {'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'stage':stage, **fields}
    write(Path(dest)/'status.json',record)
    with (Path(dest)/'events.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(record,allow_nan=False)+'\n')
    print(json.dumps(record,allow_nan=False),flush=True)
