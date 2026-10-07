"""Run-local portable file identities and atomic evidence writes."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

RUN = Path(__file__).resolve().parent
REPO = RUN.parents[1]
ARCHIVE = Path(os.environ.get('RUN063_ARCHIVE', str(RUN / 'archive/root')))


def fs(path):
    path = Path(path).absolute()
    return Path('\\\\?\\' + str(path)) if os.name == 'nt' and not str(path).startswith('\\\\?\\') else path


def read(path):
    return json.loads(fs(path).read_text(encoding='utf-8'))


def write(path, value):
    path = fs(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8', newline='\n')
    temp.replace(path)


def sha(path):
    with fs(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def record(path, root=RUN):
    path = Path(path).resolve()
    if not fs(path).is_file() or fs(path).is_symlink():
        raise ValueError(f'Regular file required: {path}')
    return {'path': path.relative_to(Path(root).resolve()).as_posix(),
            'bytes': fs(path).stat().st_size, 'sha256': sha(path)}


def verify(row, root=RUN):
    root = Path(root).resolve()
    path = (root/row['path']).resolve()
    if not path.is_relative_to(root) or record(path, root) != row:
        raise ValueError(f'Identity mismatch: {row["path"]}')
    return path


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def archived_run(number):
    paths = list((ARCHIVE/'runs').glob(f'{number:03d}-*'))
    if len(paths) != 1:
        raise ValueError(f'Expected one archived Run{number}')
    return paths[0]
