"""Run069 identities and read-only reuse of the qualified Run068 runtime."""
import hashlib
import json
import os
from pathlib import Path
import sys

RUN = Path(__file__).resolve().parent
REPO = RUN.parents[1]
SOURCE = REPO/'runs/068-2026-10-08-pythia14m-aligned-vocabulary-tiles/inference'
sys.path.insert(0, str(SOURCE))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8', newline='\n')
    temporary.replace(path)


def record(path):
    path = Path(path).resolve()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': path.relative_to(REPO).as_posix(), 'bytes': path.stat().st_size, 'sha256': digest}


def verify(row):
    path = (REPO/row['path']).resolve()
    if not path.is_relative_to(REPO) or path.is_symlink() or record(path) != row:
        raise ValueError('Identity mismatch: '+row['path'])
    return path


def config():
    cfg = read(RUN/'config.json')
    if cfg['new_training'] or len({v['name'] for v in cfg['variants']}) != len(cfg['variants']):
        raise ValueError('Frozen-checkpoint diagnostic and unique variants required')
    old = read(SOURCE/'config.json')
    if cfg['numerical_bounds'] != old['numerical_bounds'] or cfg['validation'] != old['validation']:
        raise ValueError('Keep source numerical bounds and complete validation unchanged')
    for v in cfg['variants']:
        if v['n'] not in (128,256,512) or v['groups'] not in (1,2):
            raise ValueError('Unsupported consumer tile')
        if v['family'] not in ('legacy','direct','prefetch','reuse'):
            raise ValueError('Unknown implementation family')
        if v['family']=='reuse' and v['groups']!=2:
            raise ValueError('Independent two-mask reuse required')
        if v['family']!='reuse' and v['groups']!=1:
            raise ValueError('Single-mask variant required')
        if v['family'] in ('prefetch','reuse') and v['n']==512:
            raise ValueError('Bound the register-heavy variants to N128/N256')
    return cfg


def sources():
    return read(SOURCE/'provenance/trained-inputs.json')


def runtime(seed):
    import torch
    from io_utils import configure_cuda
    configure_cuda(seed)
    torch.set_num_threads(4)


def load_model(checkpoint):
    from io_utils import load_model as source_load
    return source_load(checkpoint)


def manifest_identity(development=False):
    inventory = RUN/'provenance/upload-inventory.json'
    rows = read(inventory)
    for row in rows:
        verify(row)
    commit = os.environ.get('RUN069_SOURCE_COMMIT')
    if not development and (not commit or len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit)):
        raise ValueError('Verified source commit required')
    return {'commit': commit, 'inventory': record(inventory), 'development': development}


def support(value, threshold):
    import torch
    # Value is native FP32 pre-gate input. The certificate matches the post-gate
    # certificate because rejected corners are zero and kappa is nonnegative.
    live = value.reshape(-1,128)[::16,::16] > threshold
    return (live.int()*(1 << torch.arange(8, device=value.device))).sum(-1).int()


def shortlist(timings, limit):
    eligible = [v for v in timings if v != 'legacy_n256']
    # Selection is exploratory. Fresh final processes measure the chosen modes.
    winners = []
    for mode in ('no_skip','sparse'):
        best = min(eligible, key=lambda v: timings[v][mode]['cuda_ms']['geomean'])
        if best not in winners:
            winners.append(best)
    return winners[:limit]


def emit(destination, stage, **fields):
    from datetime import datetime, timezone
    row = {'utc': datetime.now(timezone.utc).isoformat(), 'stage': stage, **fields}
    write(Path(destination)/'status.json', row)
    with (Path(destination)/'events.jsonl').open('a', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(row, allow_nan=False)+'\n')
    print(json.dumps(row, allow_nan=False), flush=True)


def check_runtime(development):
    import torch, transformers, numpy
    cfg = config()
    values = {'torch': torch.__version__.split('+')[0], 'transformers': transformers.__version__,
              'numpy': numpy.__version__, 'cuda': torch.version.cuda}
    if values != cfg['runtime']:
        raise ValueError('Runtime pins differ: '+str(values))
    values['gpu'] = torch.cuda.get_device_name()
    if not development and values['gpu'] != 'NVIDIA GeForce RTX 5090':
        raise ValueError('RTX 5090 required for accepted measurements')
    return values
