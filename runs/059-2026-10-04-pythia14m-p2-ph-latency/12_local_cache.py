"""Hash-verified local-disk cache; persistent originals remain untouched."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from io_utils import RUN, read, sha, write


def main():
    if (RUN/'artifacts/controller.lock').exists():
        raise RuntimeError('Stop the owned controller before cache setup')
    started=time.monotonic()
    local=Path('/tmp/run059-cache');local.mkdir(exist_ok=False)
    # Read the single verified upload, avoiding thousands of network metadata reads.
    import tarfile
    receipt=read(RUN/'bundles/input-001.receipt.json')
    archive=Path('/workspace/run059-input.tar')
    if archive.stat().st_size!=receipt['bytes'] or sha(archive)!=receipt['sha256']:
        raise RuntimeError('Original upload changed')
    with tarfile.open(archive) as tar:
        members=[member for member in tar.getmembers() if member.name.startswith('archive/')]
        tar.extractall(local,members=members,filter='data')
    frozen=read(RUN/'provenance/archive.json')['files']
    for entry in frozen:
        row=entry['snapshot'];path=local/row['path']
        if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:
            raise RuntimeError('Frozen source cache mismatch: '+row['path'])
    write(RUN/'artifacts/control/local-cache-status.json',{'stage':'sources_verified','files':len(frozen),'seconds':time.monotonic()-started})
    runtime=RUN/'runtime/venv';runtime_rows=[];total=0
    free=shutil.disk_usage(local).free
    def copy_verified(source,dest):
        nonlocal total
        source=Path(source);dest=Path(dest);size=source.stat().st_size
        if total+size+2*1024**3>free:raise RuntimeError('Local disk reserve exceeded')
        digest=hashlib.sha256()
        with source.open('rb') as inp,dest.open('wb') as out:
            for chunk in iter(lambda:inp.read(8*1024**2),b''):
                digest.update(chunk);out.write(chunk)
        shutil.copymode(source,dest)
        identity=digest.hexdigest()
        if dest.stat().st_size!=size or sha(dest)!=identity:raise RuntimeError('Runtime cache byte mismatch')
        runtime_rows.append({'path':source.relative_to(runtime).as_posix(),'bytes':size,'sha256':identity})
        total+=size
        return str(dest)
    shutil.copytree(runtime,local/'venv',symlinks=True,copy_function=copy_verified)
    links=[]
    for path in runtime.rglob('*'):
        if path.is_symlink():
            target=os.readlink(path);cached=local/'venv'/path.relative_to(runtime)
            if not cached.is_symlink() or os.readlink(cached)!=target:raise RuntimeError('Runtime symlink mismatch')
            links.append({'path':path.relative_to(runtime).as_posix(),'target':target})
    for source,cached in [(RUN/'archive/root',local/'archive/root'),(runtime,local/'venv')]:
        original=source.with_name(source.name+'-persistent-original')
        if source.is_symlink() or original.exists():raise RuntimeError('Cache already installed')
        source.rename(original);source.symlink_to(cached,target_is_directory=True)
    write(RUN/'artifacts/control/local-cache.json',{'status':'verified','source_files':len(frozen),
        'runtime_files':runtime_rows,'runtime_symlinks':links,'runtime_bytes':total,
        'seconds':time.monotonic()-started,'persistent_originals_preserved':True,
        'scientific_sources_or_inputs_changed':False})
    print(json.dumps({'local_cache':'verified','runtime_bytes':total,'seconds':time.monotonic()-started}),flush=True)


if __name__=='__main__':main()
