"""Snapshot code and upload the eight missing immutable checkpoints once."""
import hashlib,json,tarfile,io,time,shlex
from pathlib import Path
from transport import connect,execute
RUN=Path(__file__).resolve().parent.parent;REPO=RUN.parents[1]
def sha(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
paths=list(RUN.glob('*.py'))+list(RUN.glob('*.cu'))+[RUN/'config.json',RUN/'README.md']+list((RUN/'provenance').rglob('*.json'))+list((RUN/'provenance/originals').rglob('*.yaml'))
snapshot={p.relative_to(RUN).as_posix():p.read_bytes() for p in paths}
inputs=json.loads((RUN/'provenance/transfer-inputs.json').read_text())['files']
bundle=RUN/'prelaunch/inputs-source-001.tar.gz';rows=[]
with tarfile.open(bundle,'w:gz',compresslevel=1) as tar:
 for rel,raw in snapshot.items():
  info=tarfile.TarInfo(rel);info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
  rows.append({'path':rel,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
 for row in inputs:
  if row['path'] in snapshot:continue
  rows.append({k:row[k] for k in ('path','bytes','sha256')})
  if not row['reuse_run049']:tar.add(REPO/row['source'],arcname=row['path'],recursive=False)
receipt={'files':rows,'archive_bytes':bundle.stat().st_size,'archive_sha256':sha(bundle)}
(RUN/'prelaunch/upload-001.json').write_text(json.dumps(receipt,indent=2)+'\n')
print({'packaged_files':len(rows),'upload_bytes':bundle.stat().st_size},flush=True)
c=connect()
try:
 execute(c,'mkdir -p /workspace/run051 /workspace/run051-control')
 s=c.open_sftp();last=[time.monotonic()]
 def progress(sent,total):
  if time.monotonic()-last[0]>45:print({'sent_bytes':sent,'total_bytes':total},flush=True);last[0]=time.monotonic()
 s.put(str(bundle),'/workspace/run051-input-001.tar.gz',callback=progress)
 s.put(str(RUN/'prelaunch/upload-001.json'),'/workspace/run051-control/upload-001.json');s.close()
 code='''import pathlib,hashlib,json,tarfile,shutil
root=pathlib.Path('/workspace/run051');archive=pathlib.Path('/workspace/run051-input-001.tar.gz')
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
