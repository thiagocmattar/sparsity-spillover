"""Copy one verified final model through local storage and queue exclusive GPU timing."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

RUN=Path(__file__).resolve().parent.parent
REMOTE='/workspace/sparsity-spillover/runs/'+RUN.name
KEY=str(Path.home()/'.runpod/ssh/runpodctl-ssh-key')


def connection(node):
    s=node['ssh'];host=s['username']+'@'+s['host']
    return host,['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-i',KEY,'-p',str(s['port']),host],['scp','-q','-o','BatchMode=yes','-i',KEY,'-P',str(s['port'])]


def sha(path):
    with path.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--condition',required=True);args=p.parse_args()
    nodes=json.loads((RUN/'prelaunch/parallel-connections-002.json').read_text())
    node,=[n for n in nodes if args.condition in n['conditions']]
    host,ssh,scp=connection(node)
    folder=RUN/'prelaunch/final-transfers'/args.condition;folder.mkdir(parents=True,exist_ok=True)
    code='R='+repr(REMOTE)+'\nC='+repr(args.condition)+'\n'+'''
import pathlib,json,hashlib,tarfile
r=pathlib.Path(R);matches=[]
for p in (r/'artifacts').glob('verification-*.json'):
 d=json.loads(p.read_text())
 if d['status']=='verified':matches.extend(x for x in d['conditions'] if x['condition']['id']==C)
row,=matches
assert row['status']=='verified'
source=r/'artifacts/attempts'/row['attempt']/row['final_checkpoint']['path']
files=[]
for p in sorted(source.iterdir()):
 if p.is_file() and p.name!='training_state.pt':
  with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
  files.append(dict(path=p.name,bytes=p.stat().st_size,sha256=digest))
control=pathlib.Path('/workspace/run054-control');archive=control/(C+'-final.tar')
with tarfile.open(archive,'w') as a:
 for record in files:a.add(source/record['path'],arcname=record['path'],recursive=False)
with archive.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
catalog=dict(conditions=[dict(id=C,checkpoint=R+'/latency/incoming/'+C,files=files,training_verification=row)],
 archive=dict(bytes=archive.stat().st_size,sha256=digest))
(control/(C+'-catalog.json')).write_text(json.dumps(catalog,indent=2)+'\\n')
print(json.dumps(dict(condition=C,attempt=row['attempt'],final_loss=row['final_loss'],bytes=archive.stat().st_size)))
'''
    subprocess.run(ssh+['python3 -'],input=code,text=True,check=True)
    for name in ('final.tar','catalog.json'):
        subprocess.run(scp+[f'{host}:/workspace/run054-control/{args.condition}-{name}',str(folder/name)],check=True)
    catalog=json.loads((folder/'catalog.json').read_text());archive=folder/'final.tar'
    assert archive.stat().st_size==catalog['archive']['bytes'] and sha(archive)==catalog['archive']['sha256']
    local_model=folder/'model';local_model.mkdir(exist_ok=True)
    with tarfile.open(archive) as handle:handle.extractall(local_model,filter='data')
    for row in catalog['conditions'][0]['files']:
        path=local_model/row['path'];assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
    target,ssh,scp=connection(nodes[4])
    # Stream the model through SSH: one SFTP upload suffered a severe per-connection stall.
    with archive.open('rb') as handle:
        subprocess.run(ssh+[f'cat > /workspace/run054-control/{args.condition}-final.tar'],stdin=handle,check=True)
    subprocess.run(scp+[str(folder/'catalog.json'),f'{target}:/workspace/run054-control/{args.condition}-catalog.json'],check=True)
    code='R='+repr(REMOTE)+'\nC='+repr(args.condition)+'\n'+'''
import pathlib,json,hashlib,tarfile,subprocess,os
control=pathlib.Path('/workspace/run054-control');marker=control/(C+'-latency.pid')
if marker.exists():raise FileExistsError('Condition was already queued')
catalog=control/(C+'-catalog.json');data=json.loads(catalog.read_text());archive=control/(C+'-final.tar')
with archive.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
assert digest==data['archive']['sha256'] and archive.stat().st_size==data['archive']['bytes']
dest=pathlib.Path(data['conditions'][0]['checkpoint']);dest.mkdir(parents=True,exist_ok=False)
with tarfile.open(archive) as a:a.extractall(dest,filter='data')
env=dict(os.environ,PATH='/workspace/run054-venv/bin:'+os.environ['PATH'],OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
with (control/(C+'-latency.log')).open('x') as log:
 process=subprocess.Popen(['flock','/workspace/run054-control/latency.lock','/workspace/run054-venv/bin/python',R+'/latency/03_final_grid.py','--catalog',str(catalog),'--condition',C,'--tag','final-001'],
  env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
marker.write_text(str(process.pid)+'\\n');print(json.dumps(dict(condition=C,pid=process.pid,status='queued')))
'''
    subprocess.run(ssh+['python3 -'],input=code,text=True,check=True)
    (folder/'transfer-verified.json').write_text(json.dumps(dict(status='verified_and_queued',condition=args.condition,
        source_pod=node['id'],latency_pod=nodes[4]['id'],archive=catalog['archive']),indent=2)+'\n')


if __name__=='__main__':main()
