"""Losslessly compress the already verified transfer archive for replacement hosts."""
import gzip,hashlib,json,pathlib,shutil,time
run=pathlib.Path(__file__).resolve().parent.parent
source=run.parent/'054-2026-09-24-pythia31m-t2-ph/prelaunch/input-data.tar'
target=run/'prelaunch/input-data.tar.gz'
started=time.time()
with source.open('rb') as f:
    assert hashlib.file_digest(f,'sha256').hexdigest()=='86415e9b8b3f32f4a0fbfb210d37fb4bd1fea66442a15b311c1f44ff2b2652b5'
with source.open('rb') as f,target.open('xb') as out:
    with gzip.GzipFile(fileobj=out,mode='wb',compresslevel=1,mtime=0) as compressed:
        shutil.copyfileobj(f,compressed,8*1024*1024)
with target.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
receipt=dict(status='compressed',bytes=target.stat().st_size,sha256=digest,seconds=time.time()-started)
(run/'prelaunch/compressed-cache.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
