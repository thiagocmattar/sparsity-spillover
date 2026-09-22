"""Upload the explicit Run050 source/input inventory to the retained Pod."""
import hashlib,json,posixpath,sys,tarfile,io,time
from pathlib import Path
from transport import connect,execute

RUN=Path(__file__).resolve().parent.parent
tag=f'{len(list((RUN/"prelaunch").glob("upload-*.json")))+1:03d}'
paths=list(RUN.glob('*.py'))+list(RUN.glob('*.cu'))+[RUN/'config.json',RUN/'README.md']
paths+=list((RUN/'provenance').glob('*.json'))+list((RUN/'inputs').glob('*'))
snapshot={p.relative_to(RUN).as_posix():p.read_bytes() for p in paths}
client=connect()
try:
 execute(client,'mkdir -p /workspace/run050/provenance /workspace/run050/inputs /workspace/run050-control')
 s=client.open_sftp();files=[]
 for rel,raw in snapshot.items():
  remote='/workspace/run050/'+rel
  digest=hashlib.sha256(raw).hexdigest();existing=None
  try:
   with s.file(remote,'rb') as f:existing=hashlib.sha256(f.read()).hexdigest()
  except FileNotFoundError:pass
  if existing!=digest:
   if existing is not None and (rel.startswith('inputs/') or rel=='provenance/inputs.json'):
    raise RuntimeError('Immutable input differs: '+rel)
   temporary=remote+'.upload-'+tag
   with s.file(temporary,'wb') as f:f.write(raw)
   s.posix_rename(temporary,remote)
  files.append({'path':rel,'bytes':len(raw),'sha256':digest})
 receipt=RUN/f'prelaunch/upload-{tag}.json';receipt.write_text(json.dumps({'files':files},indent=2)+'\n')
 (RUN/'provenance/source-history').mkdir(exist_ok=True)
 with tarfile.open(RUN/f'provenance/source-history/upload-{tag}.tar.gz','w:gz') as archive:
  for rel,raw in snapshot.items():
   info=tarfile.TarInfo(rel);info.size=len(raw);info.mtime=int(time.time());archive.addfile(info,io.BytesIO(raw))
 s.put(str(receipt),f'/workspace/run050-control/upload-{tag}.json');s.close()
 print(execute(client,"python3 -c \"import pathlib,json,hashlib; r=pathlib.Path('/workspace/run050');d=json.load(open('/workspace/run050-control/upload-"+tag+".json'));assert all(hashlib.sha256((r/x['path']).read_bytes()).hexdigest()==x['sha256'] for x in d['files']);print('Verified uploaded files:',len(d['files']))\""))
finally:client.close()
