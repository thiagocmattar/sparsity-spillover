"""Bundle only completed attempts, copy once, verify every extracted file."""
import argparse,hashlib,json,shlex,tarfile
from pathlib import Path
from transport import connect,execute
p=argparse.ArgumentParser();p.add_argument('attempts',nargs='+');a=p.parse_args()
run=Path(__file__).resolve().parent.parent;dest=run/'retrieval';dest.mkdir(exist_ok=True)
code='''import pathlib,hashlib,json,tarfile,sys
a=sys.argv[1];root=pathlib.Path('/workspace/run050/artifacts')/a
result=root/'result.json'
assert result.exists() and json.loads(result.read_text())['status']!='running','Completed result required'
paths=sorted(p for p in root.rglob('*') if p.is_file())
rows=[{'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]
out=pathlib.Path('/workspace/run050-transfer');out.mkdir(exist_ok=True)
manifest=out/(a+'.json');manifest.write_text(json.dumps({'attempt':a,'files':rows},indent=2)+'\\n')
with tarfile.open(out/(a+'.tar.gz'),'w:gz',compresslevel=3) as t:
 for p in paths:t.add(p,arcname=p.relative_to(root).as_posix(),recursive=False)
print(json.dumps({'files':len(rows),'bytes':sum(r['bytes'] for r in rows)}))
'''
c=connect()
try:
 s=c.open_sftp()
 for attempt in a.attempts:
  assert attempt.replace('-','').isalnum()
  print(attempt,execute(c,'python3 -c '+shlex.quote(code)+' '+shlex.quote(attempt),timeout=180).strip(),flush=True)
  for suffix in ('json','tar.gz'):s.get(f'/workspace/run050-transfer/{attempt}.{suffix}',str(dest/f'{attempt}.{suffix}'))
  manifest=json.loads((dest/f'{attempt}.json').read_text());target=(run/'artifacts'/attempt).resolve();target.mkdir(parents=True,exist_ok=True)
  with tarfile.open(dest/f'{attempt}.tar.gz','r:gz') as t:
   for member in t.getmembers():
    assert member.isfile(),'Only plain files may be transferred'
    path=(target/member.name).resolve();assert path.is_relative_to(target)
    path.parent.mkdir(parents=True,exist_ok=True)
    with t.extractfile(member) as f:path.write_bytes(f.read())
  for row in manifest['files']:
   path=target/row['path'];assert path.stat().st_size==row['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
  for suffix in ('log','exit','pid','sh'):
   try:s.get(f'/workspace/run050-control/{attempt}.{suffix}',str(run/'prelaunch'/f'{attempt}.{suffix}'))
   except FileNotFoundError:pass
  print('Verified',len(manifest['files']),'files',flush=True)
 s.close()
finally:c.close()
