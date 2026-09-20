"""Copy identical runtime bytes to host-local storage before final measurements."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT=Path('/workspace/run045')
CONTROL=Path('/workspace/run045-control')
LOCAL=Path('/tmp/run045-local-runtime')
PIPELINE_PID=475
DEADLINE=1789941035


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    started=time.monotonic()
    # Preflight is allowed to finish; its parent shell is stopped before final.
    while (ROOT/'artifacts/controller.lock').exists():
        if time.time()>DEADLINE-2400:raise TimeoutError('Recovery reserve required')
        time.sleep(10)
    summary=json.loads((ROOT/'artifacts/preflight/summary-001.json').read_text())
    assert len(summary)==3 and all(r['qualified'] and r['status']=='complete' for r in summary)
    assert shutil.disk_usage('/tmp').free > 10*1024**3
    LOCAL.mkdir(exist_ok=True)
    files=[]
    for name in ('venv','extensions'):
        source=ROOT/'runtime'/name
        print('enumerating',name,flush=True)
        for p in source.rglob('*'):
            target=LOCAL/name/p.relative_to(source)
            target.parent.mkdir(parents=True,exist_ok=True)
            if p.is_symlink():
                if not target.is_symlink():target.symlink_to(os.readlink(p),target_is_directory=p.is_dir())
                assert target.is_symlink() and os.readlink(p)==os.readlink(target)
            elif p.is_file():
                files.append((p,target))
    def copy_verify(pair):
        source,target=pair
        before=source.stat()
        if not target.exists() or (target.stat().st_size,target.stat().st_mtime_ns)!=(before.st_size,before.st_mtime_ns):
            shutil.copy2(source,target)
        digest=sha(source)
        assert sha(target)==digest,str(source)
        return {'path':str(source.relative_to(ROOT/'runtime')),'bytes':before.st_size,'sha256':digest}
    records=[]
    print('copy_and_verify',len(files),'files, 8 bounded I/O workers',flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(copy_verify,pair) for pair in files]):
            records.append(future.result())
            if len(records)%1000==0:print('verified_files',len(records),'of',len(files),flush=True)
    records.sort(key=lambda r:r['path'])
    for name in ('venv','extensions'):
        source=ROOT/'runtime'/name
        original=ROOT/'runtime'/(name+'-network-original')
        assert not original.exists()
        source.rename(original)
        source.symlink_to(LOCAL/name,target_is_directory=True)
    result=dict(status='verified',files=len(records),bytes=sum(r['bytes'] for r in records),
                seconds=time.monotonic()-started,kind='Infrastructure-only filesystem cache; identical runtime and extension bytes.',
                persistent_originals_preserved=True,files_sha256=records)
    (CONTROL/'local-runtime-cache.json').write_text(json.dumps(result,indent=2)+'\n')
    cmd=Path(f'/proc/{PIPELINE_PID}/cmdline').read_bytes().replace(b'\x00',b' ').strip()
    assert cmd==b'bash /workspace/run045-control/pipeline-001.sh'
    os.kill(PIPELINE_PID,signal.SIGCONT)
    print(json.dumps({k:v for k,v in result.items() if k!='files_sha256'}),flush=True)


if __name__=='__main__':main()
