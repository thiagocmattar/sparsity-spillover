"""Run-owned paths and durable progress records."""
import hashlib
import json
from pathlib import Path
import time

RUN = Path(__file__).resolve().parent

def read(path):
    return Path(path).read_text(encoding='utf-8')

def load(path):
    return json.loads(read(path))

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def event(dest, stage, **fields):
    record = {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'stage': stage, **fields}
    write(dest / 'status.json', record)
    with (dest / 'events.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, allow_nan=False) + '\n')
    print(json.dumps(record), flush=True)
