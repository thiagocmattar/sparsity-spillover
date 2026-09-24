"""Copy only the two closed seed-Pod attempts while its third condition trains."""
from pathlib import Path
import hashlib,json,subprocess,tarfile,shutil
from datetime import datetime,timezone
RUN=Path(__file__).resolve().parent.parent
KEY=str(Path.home()/'.runpod/ssh/runpodctl-ssh-key')
SSH=['ssh','-o','BatchMode=yes','-i',KEY,'-p','27548','root@103.196.86.181']
SCP=['scp','-q','-o','BatchMode=yes','-i',KEY,'-P','27548']
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
code='''import pathlib,json,hashlib,tarfile
r=pathlib.Path('/workspace/sparsity-spillover/runs/055-2026-09-24-pythia31m-t7-ph');c=pathlib.Path('/workspace/run055-control')
assert json.loads((r/'artifacts/pipeline/parallel-0-001/status.json').read_text())['status']=='completed'
files=set();verified=[];wanted={'a7-h-ol1-kappa-0','a7-h-ol1-kappa-0p01'}
for v in (r/'artifacts').glob('verification-*.json'):
 d=json.loads(v.read_text())
 if d['status']=='verified' and {row['condition']['id'] for row in d['conditions']}<=wanted:
  files.add(v);verified.extend(d['conditions'])
assert len(verified)==2 and {row['condition']['id'] for row in verified}==wanted
for row in verified:
 p=r/'artifacts/attempts'/row['attempt'];assert json.loads((p/'manifest.json').read_text())['status']=='completed'
 files.update(f for f in p.rglob('*') if f.is_file())
rows=[]
for p in sorted(files):
 with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
 rows.append(dict(path=p.relative_to(r).as_posix(),bytes=p.stat().st_size,sha256=digest))
archive=c/'completed-seed-001.tar'
with tarfile.open(archive,'x') as a:
 for row in rows:a.add(r/row['path'],arcname=row['path'],recursive=False)
with archive.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
receipt=dict(files=rows,archive=dict(bytes=archive.stat().st_size,sha256=digest))
(c/'completed-seed-001.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(dict(files=len(rows),bytes=archive.stat().st_size)))
'''
subprocess.run(SSH+['python3 -'],input=code,text=True,check=True)
folder=RUN/'prelaunch/retrievals/completed-seed-001';folder.mkdir(parents=True,exist_ok=True)
for suffix in ['json','tar']:
    subprocess.run(SCP+['root@103.196.86.181:/workspace/run055-control/completed-seed-001.'+suffix,str(folder/('evidence.'+suffix))],check=True)
receipt=json.loads((folder/'evidence.json').read_text());archive=folder/'evidence.tar'
assert archive.stat().st_size==receipt['archive']['bytes'] and sha(archive)==receipt['archive']['sha256']
dest=folder/'files';dest.mkdir(exist_ok=True)
with tarfile.open(archive) as a:a.extractall(dest,filter='data')
for row in receipt['files']:
    source=(dest/row['path']).resolve();target=(RUN/row['path']).resolve()
    assert source.is_relative_to(dest.resolve()) and target.is_relative_to(RUN.resolve())
    assert source.stat().st_size==row['bytes'] and sha(source)==row['sha256']
    if target.exists():assert target.stat().st_size==row['bytes'] and sha(target)==row['sha256']
    else:target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
out=dict(status='verified',pod='rjtpxhwiek5tpe',scope='Two completed attempts only; third attempt remains active',utc=datetime.now(timezone.utc).isoformat(),files=len(receipt['files']),bytes=sum(x['bytes'] for x in receipt['files']),archive=receipt['archive'])
(RUN/'prelaunch/retrieved-completed-seed-001.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
