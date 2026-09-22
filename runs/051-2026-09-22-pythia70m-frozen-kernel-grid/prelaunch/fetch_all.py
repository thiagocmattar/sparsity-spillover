"""Retrieve terminal artifacts and control logs in one hash-verified bundle."""
import hashlib,json,shlex,subprocess,tarfile
from pathlib import Path
from transport import connect,execute
RUN=Path(__file__).resolve().parent.parent
def sha(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
code='''import pathlib,hashlib,json,tarfile
root=pathlib.Path('/workspace/run051');out=pathlib.Path('/workspace/run051-transfer');out.mkdir(exist_ok=True)
assert json.load(open(root/'artifacts/final-001/result.json'))['status']=='complete'
paths=[(p,p.relative_to(root).as_posix()) for p in sorted((root/'artifacts').rglob('*')) if p.is_file()]
control=pathlib.Path('/workspace/run051-control')
paths += [(p,'prelaunch/'+p.name) for p in sorted(control.iterdir()) if p.is_file()]
rows=[]
for p,rel in paths:
 with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
 rows.append({'path':rel,'bytes':p.stat().st_size,'sha256':digest})
archive=out/'all-terminal-001.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=3) as t:
 for p,rel in paths:t.add(p,arcname=rel,recursive=False)
with archive.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
manifest={'files':rows,'archive_bytes':archive.stat().st_size,'archive_sha256':digest}
(out/'all-terminal-001.json').write_text(json.dumps(manifest,indent=2)+'\\n')
print(json.dumps({'files':len(rows),'bytes':sum(r['bytes'] for r in rows),'archive_bytes':manifest['archive_bytes']}))
'''
c=connect()
try:
 print(execute(c,'python3 -c '+shlex.quote(code),timeout=300),flush=True)
 dest=RUN/'retrieval';dest.mkdir(exist_ok=True)
 s=c.open_sftp();s.get('/workspace/run051-transfer/all-terminal-001.json',str(dest/'all-terminal-001.json'));s.close()
 info_root=RUN.parent/'049-2026-09-22-pythia70m-short-row-limits'/'prelaunch'
 info=json.loads((info_root/'ssh-001.json').read_text())
 command=['C:/Windows/System32/OpenSSH/ssh.exe','-i',info['ssh_key']['path'],'-p',str(info['port']),'-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+str((info_root/'known_hosts').resolve()),'root@'+info['ip'],'cat /workspace/run051-transfer/all-terminal-001.tar.gz']
 archive=dest/'all-terminal-001.tar.gz'
 with archive.open('wb') as f:subprocess.run(command,stdout=f,check=True,timeout=300)
 manifest=json.loads((dest/'all-terminal-001.json').read_text())
 assert archive.stat().st_size==manifest['archive_bytes'] and sha(archive)==manifest['archive_sha256']
 allowed={row['path'] for row in manifest['files']};seen=set()
 with tarfile.open(archive) as t:
  for member in t.getmembers():
   assert member.isfile() and member.name in allowed and member.name not in seen
   path=(RUN/member.name).resolve();assert path.is_relative_to(RUN.resolve())
   path.parent.mkdir(parents=True,exist_ok=True)
   with t.extractfile(member) as f:path.write_bytes(f.read())
   seen.add(member.name)
 assert seen==allowed
 for row in manifest['files']:
  path=RUN/row['path'];assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],row['path']
 print({'verified_files':len(seen),'verified_bytes':sum(r['bytes'] for r in manifest['files'])},flush=True)
finally:c.close()
