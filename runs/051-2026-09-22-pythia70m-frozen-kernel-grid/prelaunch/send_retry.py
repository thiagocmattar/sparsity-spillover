"""Retry the verified bundle through local disk, retaining the interrupted upload."""
import json,shlex,time
from pathlib import Path
from transport import connect,execute
RUN=Path(__file__).resolve().parent.parent
bundle=RUN/'prelaunch/inputs-source-001.tar.gz'
receipt=json.loads((RUN/'prelaunch/upload-001.json').read_text())
assert bundle.stat().st_size==receipt['archive_bytes']
c=connect()
try:
 execute(c,'mkdir -p /workspace/run051 /workspace/run051-control')
 s=c.open_sftp();last=[time.monotonic()]
 def progress(sent,total):
  if time.monotonic()-last[0]>45:print({'sent_bytes':sent,'total_bytes':total},flush=True);last[0]=time.monotonic()
 s.put(str(bundle),'/tmp/run051-input-001.tar.gz',callback=progress)
 s.put(str(RUN/'prelaunch/upload-001.json'),'/workspace/run051-control/upload-001.json');s.close()
 code='''import pathlib,hashlib,json,tarfile,shutil
root=pathlib.Path('/workspace/run051');archive=pathlib.Path('/tmp/run051-input-001.tar.gz')
receipt=json.load(open('/workspace/run051-control/upload-001.json'))
with archive.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==receipt['archive_sha256']
with tarfile.open(archive) as t:
 for m in t.getmembers():assert m.isfile() and (root/m.name).resolve().is_relative_to(root)
 t.extractall(root,filter='data')
inputs=json.load(open(root/'provenance/transfer-inputs.json'))['files']
for row in inputs:
 if row['reuse_run049'] and not (root/row['path']).exists():
  dest=root/row['path'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(pathlib.Path('/workspace/run049')/row['path'],dest)
for row in receipt['files']:
 path=root/row['path'];assert path.stat().st_size==row['bytes'],row['path']
 with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==row['sha256'],row['path']
print(json.dumps({'verified_files':len(receipt['files']),'bytes':sum(r['bytes'] for r in receipt['files'])}))
'''
 print(execute(c,'python3 -c '+shlex.quote(code),timeout=600),flush=True)
finally:c.close()
