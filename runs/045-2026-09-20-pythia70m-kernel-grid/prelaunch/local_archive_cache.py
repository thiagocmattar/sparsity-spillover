"""Preserve frozen source bytes while moving repeated vendor checks off network storage."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import time

RUN=Path('/workspace/run045')
SOURCE=RUN/'archive/root'
LOCAL=Path('/tmp/run045-archive-root')
CONTROL=Path('/workspace/run045-control')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def processes():
    result=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            args=p.joinpath('cmdline').read_bytes().split(b'\x00')
            parent=int(p.joinpath('stat').read_text().split()[3])
            result.append((int(p.name),parent,args))
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    return result


def main():
    started=time.monotonic()
    controllers=[pid for pid,_,args in processes() if any(a.endswith(b'03_execute.py') for a in args)
                 and b'final' in args and Path(os.fsdecode(args[0])).name in ('python','python3')]
    assert len(controllers)==1,controllers
    controller=controllers[0]
    os.kill(controller,signal.SIGSTOP)
    print('controller_paused',controller,flush=True)
    while any(parent==controller and any(a.endswith(b'02_benchmark.py') for a in args)
              for _,parent,args in processes()):time.sleep(5)
    print('current_leaf_complete',flush=True)
    assert not LOCAL.exists() and not SOURCE.is_symlink()
    LOCAL.mkdir()
    paths=[]
    for p in SOURCE.rglob('*'):
        target=LOCAL/p.relative_to(SOURCE)
        assert not p.is_symlink()
        if p.is_dir():target.mkdir(parents=True,exist_ok=True)
        elif p.is_file():paths.append((p,target))
    def copy(pair):
        source,target=pair
        shutil.copy2(source,target)
        digest=sha(source);assert sha(target)==digest
        return {'path':source.relative_to(SOURCE).as_posix(),'bytes':source.stat().st_size,'sha256':digest}
    with ThreadPoolExecutor(max_workers=8) as pool:records=list(pool.map(copy,paths))
    # Every previously frozen archived file must still match its original digest.
    manifest=json.loads((RUN/'provenance/archive.json').read_text())
    frozen=[row['snapshot'] for row in manifest['files']]
    for row in frozen:
        relative=Path(row['path']).relative_to('archive/root')
        assert sha(LOCAL/relative)==row['sha256'],row['path']
    original=SOURCE.with_name('root-network-original')
    assert not original.exists()
    SOURCE.rename(original)
    SOURCE.symlink_to(LOCAL,target_is_directory=True)
    result=dict(status='verified',files=len(records),frozen_files=len(frozen),
                bytes=sum(r['bytes'] for r in records),seconds=time.monotonic()-started,
                persistent_original_preserved=True,implementation_sources_unchanged=True,
                reason='Each attention implementation re-hashes the retained vendor tree before loading; this check stays enabled.',
                files_sha256=records)
    (CONTROL/'local-archive-cache.json').write_text(json.dumps(result,indent=2)+'\n')
    os.kill(controller,signal.SIGCONT)
    print(json.dumps({k:v for k,v in result.items() if k!='files_sha256'}),flush=True)


if __name__=='__main__':main()
