"""Make a small, real Git deployment snapshot without unrelated workspace edits."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]


def git(*args,cwd=ROOT):
    return subprocess.check_output(['git',*args],cwd=cwd,text=True).strip()


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    if not a.tag.isalnum():raise ValueError('Alphanumeric tag required')
    stage=ROOT/'tmp'/('run043-deployment-'+a.tag)
    stage.mkdir(exist_ok=False)
    paths=[ROOT/'pyproject.toml',ROOT/'README.md',ROOT/'.gitignore']
    paths+=list((ROOT/'src/sparsity_research').glob('*.py'))
    paths+=list((ROOT/'runs/004-2026-08-29-pythia14m-full-pass-l1n').glob('*.py'))
    paths+=[ROOT/'runs/034-2026-09-17-pythia70m-h-only-ol1/artifacts/verification.json']
    for folder in ('017-2026-09-01-pythia70m-selected-ladder-portable-init','018-2026-09-01-pythia70m-selected-ladder-canonical-init'):
        paths+=list((ROOT/'runs'/folder).glob('*.py'))
    paths+=[p for p in RUN.iterdir() if p.is_file() and p.suffix in {'.py','.sh','.ps1','.yaml','.md'}]
    paths+=[RUN/'.gitignore']
    paths+=[RUN/'prelaunch/initialization/metadata.json', RUN/'architecture_config.json',RUN/'.gitattributes']
    source=[]
    for path in sorted(set(paths)):
        rel=path.relative_to(ROOT);target=stage/rel;target.parent.mkdir(parents=True,exist_ok=True)
        payload=path.read_bytes().replace(b'\r\n',b'\n');target.write_bytes(payload)
        source.append({'path':rel.as_posix(),'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()})
    origin=git('rev-parse','HEAD')
    (stage/'deployment-origin.json').write_text(json.dumps({'origin_commit':origin,'source_files':source},indent=2)+'\n')
    git('init',cwd=stage)
    git('config','user.name',git('config','user.name'),cwd=stage)
    git('config','user.email',git('config','user.email'),cwd=stage)
    git('add','.',cwd=stage)
    git('commit','-m','Capture Run043 deployment source',cwd=stage)
    bundle=RUN/'prelaunch'/('source-'+a.tag+'.bundle')
    git('bundle','create',str(bundle),'HEAD',cwd=stage)
    receipt={'origin_commit':origin,'deployment_commit':git('rev-parse','HEAD',cwd=stage),
        'path':bundle.relative_to(RUN).as_posix(),'bytes':bundle.stat().st_size,
        'sha256':hashlib.sha256(bundle.read_bytes()).hexdigest(),'source_files':source}
    (RUN/'prelaunch'/('source-'+a.tag+'.receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='source_files'}))


if __name__=='__main__':main()
