"""Build a small, raw-byte-preserving source archive from committed scoped files."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def main():
    paths=[ROOT/'pyproject.toml',ROOT/'README.md']
    paths+=list((ROOT/'src/sparsity_research').glob('*.py'))
    for name in ['004-2026-08-29-pythia14m-full-pass-l1n',
                 '017-2026-09-01-pythia70m-selected-ladder-portable-init',
                 '018-2026-09-01-pythia70m-selected-ladder-canonical-init',HERE.name]:
        paths += [p for p in (ROOT/'runs'/name).iterdir() if p.is_file() and p.suffix in {'.py','.json','.yaml','.sh','.md'}]
    paths += [HERE/'prelaunch/initialization/metadata.json',HERE/'prelaunch/pip-freeze-source.txt']
    paths += [ROOT/f'data/tokenized/minipile-pythia-14m-full/{split}/metadata.json' for split in ['train','validation']]
    files=[]
    for path in sorted(set(paths)):
        raw=path.read_bytes()
        files.append({'path':path.relative_to(ROOT).as_posix(),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
    receipt={'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'files':files,
             'deployment':'scoped_source_archive_not_a_full_git_checkout; raw bytes preserved'}
    target=HERE/'prelaunch/source-receipt.json'
    target.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8',newline='\n')
    archive=HERE/'prelaunch/source.tar.gz'
    with tarfile.open(archive,'w:gz') as t:
        for path in sorted(set(paths)|{target}):t.add(path,arcname=path.relative_to(ROOT).as_posix(),recursive=False)
    print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files':len(files),'git_commit':receipt['git_commit']}))


if __name__=='__main__':main()
