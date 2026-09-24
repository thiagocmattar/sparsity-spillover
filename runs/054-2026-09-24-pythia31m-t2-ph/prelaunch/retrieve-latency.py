"""Seal and retrieve the completed six-model timing cohort from its dedicated GPU."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import shutil
from datetime import datetime,timezone

RUN=Path(__file__).resolve().parent.parent
REMOTE='/workspace/sparsity-spillover/runs/'+RUN.name
KEY=str(Path.home()/'.runpod/ssh/runpodctl-ssh-key')


def sha(path):
    with path.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def main():
    node=json.loads((RUN/'prelaunch/parallel-connections-002.json').read_text())[4]
    s=node['ssh'];host=s['username']+'@'+s['host']
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-i',KEY,'-p',str(s['port']),host]
    scp=['scp','-q','-o','BatchMode=yes','-i',KEY,'-P',str(s['port'])]
    code='R='+repr(REMOTE)+'\n'+'''
import pathlib,json,hashlib,tarfile,sys
r=pathlib.Path(R);sys.path.insert(0,R)
from run_config import load_config,condition_specs
grids=[json.loads(p.read_text()) for p in (r/'latency/artifacts').glob('final-*-grid.json')]
assert len(grids)==6 and {g['condition'] for g in grids}=={c['id'] for c in condition_specs(load_config())}
assert all(g['status']=='completed' and len(g['replicates'])==3 and all(x['qualified'] and x['returncode']==0 for x in g['replicates']) for g in grids)
control=pathlib.Path('/workspace/run054-control');files={}
for folder in ('latency/artifacts','latency/logs'):
 for p in (r/folder).rglob('*'):
  if p.is_file():files[p.relative_to(r).as_posix()]=p
files['prelaunch/cloud-latency-component-check.json']=r/'latency/component-check.json'
for p in control.iterdir():
 if p.is_file() and (p.suffix in ('.json','.log') or p.name=='pip-freeze.txt'):
  files['prelaunch/cloud-controls/latency-001/'+p.name]=p
rows=[]
for name,p in sorted(files.items()):
 with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
 rows.append(dict(path=name,bytes=p.stat().st_size,sha256=digest))
archive=control/'final-latency-evidence.tar'
with tarfile.open(archive,'w') as a:
 for row in rows:a.add(files[row['path']],arcname=row['path'],recursive=False)
with archive.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
receipt=dict(files=rows,total_bytes=sum(row['bytes'] for row in rows),archive=dict(bytes=archive.stat().st_size,sha256=digest))
(control/'final-latency-receipt.json').write_text(json.dumps(receipt,indent=2)+'\\n')
print(json.dumps(dict(files=len(rows),bytes=receipt['total_bytes'])))
'''
    subprocess.run(ssh+['/workspace/run054-venv/bin/python -'],input=code,text=True,check=True)
    folder=RUN/'prelaunch/retrievals/latency-001';folder.mkdir(parents=True,exist_ok=True)
    for name in ('evidence.tar','receipt.json'):
        subprocess.run(scp+[f'{host}:/workspace/run054-control/final-latency-{name}',str(folder/name)],check=True)
    receipt=json.loads((folder/'receipt.json').read_text());archive=folder/'evidence.tar'
    assert archive.stat().st_size==receipt['archive']['bytes'] and sha(archive)==receipt['archive']['sha256']
    extracted=folder/'files';extracted.mkdir(exist_ok=True)
    with tarfile.open(archive) as handle:handle.extractall(extracted,filter='data')
    for row in receipt['files']:
        source=(extracted/row['path']).resolve();target=(RUN/row['path']).resolve()
        assert source.is_relative_to(extracted.resolve()) and target.is_relative_to(RUN.resolve())
        assert source.stat().st_size==row['bytes'] and sha(source)==row['sha256'],row['path']
        if target.exists():assert target.stat().st_size==row['bytes'] and sha(target)==row['sha256'],row['path']
        else:target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    result=dict(status='verified',pod=node['id'],files=len(receipt['files']),bytes=receipt['total_bytes'],
        archive=receipt['archive'],utc=datetime.now(timezone.utc).isoformat())
    (RUN/'prelaunch/retrieved-latency-001.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
