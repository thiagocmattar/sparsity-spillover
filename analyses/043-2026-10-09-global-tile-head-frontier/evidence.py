"""Small file-identity helpers local to Analysis043."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCES = {}


def identity(path):
    path = Path(path).resolve()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    row = {'path': path.relative_to(ROOT).as_posix(), 'bytes': path.stat().st_size, 'sha256': digest}
    SOURCES[row['path']] = row
    return row


def read(path):
    path = ROOT / path
    identity(path)
    return json.loads(path.read_text(encoding='utf-8'))


def verify(row, base=ROOT):
    actual = identity(base / row['path'])
    assert actual['sha256'] == row['sha256'], actual['path']
    if 'bytes' in row:
        assert actual['bytes'] == row['bytes'], actual['path']


def run(prefix):
    matches = list((ROOT / 'runs').glob(prefix + '-*'))
    assert len(matches) == 1
    return matches[0]


def write(name, value):
    path = HERE / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def dominates(a, b):
    return (a['loss'] <= b['loss'] and a['latency_ms'] <= b['latency_ms']
            and (a['loss'] < b['loss'] or a['latency_ms'] < b['latency_ms']))


def frontier(points):
    return sorted([p for p in points if not any(dominates(q, p) for q in points)],
                  key=lambda p: (p['loss'], p['latency_ms'], p['id']))
