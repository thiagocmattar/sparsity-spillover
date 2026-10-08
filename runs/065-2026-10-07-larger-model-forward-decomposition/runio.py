"""Run-local identities, independent of archived modules with identical names."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
DEPS = Path(os.environ.get('RUN065_DEPS', str(ROOT/'dependencies/repository')))

def fs(path):
    p = str(Path(path).absolute())
    return Path('\\\\?\\'+p) if os.name=='nt' and not p.startswith('\\\\?\\') else Path(p)

def read(path): return json.loads(fs(path).read_text(encoding='utf-8'))
def write(path, value):
    p=fs(path);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');t.replace(p)
def record(path,root=ROOT):
    p=Path(path).resolve()
    with fs(p).open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
    return {'path':p.relative_to(Path(root).resolve()).as_posix(),'bytes':fs(p).stat().st_size,'sha256':digest}
def verify(row,root=ROOT):
    p=(Path(root)/row['path']).resolve()
    if not p.is_relative_to(Path(root).resolve()) or record(p,root)!=row:raise ValueError('Identity mismatch '+row['path'])
    return p
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m
