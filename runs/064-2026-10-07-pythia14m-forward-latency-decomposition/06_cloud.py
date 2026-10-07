"""Scoped SSH ingress/status/retrieval with archive and individual hash checks."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
import time
import paramiko

ROOT=Path(__file__).parent;REPO=ROOT.parents[1];CLI=REPO/'tmp/runpodctl-v2.14.0.exe'
REMOTE='/workspace/run064';CONTROL='/workspace/run064-control'
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')
def sha(path):return hashlib.file_digest(Path(path).open('rb'),'sha256').hexdigest()
def execute(client,cmd,timeout=60):
    _,out,err=client.exec_command(cmd,timeout=timeout);a=out.read().decode(errors='replace');b=err.read().decode(errors='replace')
    if out.channel.recv_exit_status():raise RuntimeError(a[-2000:]+b[-2000:])
    return a+b
def connect():
    lease=read(ROOT/'prelaunch/lease-001.json')
    pod=json.loads(subprocess.check_output([str(CLI),'pod','get',lease['id']],text=True))
    if pod['id']!=lease['id'] or pod['name']!=lease['name'] or not pod['name'].startswith('run064-'):raise ValueError('Unscoped Pod')
    info=pod['ssh']
    if not info.get('ip'):
        info=json.loads(subprocess.check_output([str(CLI),'ssh','info',lease['id'],'--verbose'],text=True))
    client=paramiko.SSHClient();known=ROOT/'prelaunch/known_hosts';known.touch(exist_ok=True)
    client.load_host_keys(str(known));client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(info['ip'],port=info['port'],username='root',key_filename=[str(Path.home()/'.ssh/id_ed25519_runpod'),str(Path.home()/'.runpod/ssh/runpodctl-ssh-key')],look_for_keys=False,allow_agent=False,timeout=20,auth_timeout=20,banner_timeout=20)
    client.get_transport().set_keepalive(20);return client,lease
def prepare(client,lease):
    receipt=read(ROOT/'prelaunch/bundle-receipt.json');source=ROOT/'bundles/input-001.tar.gz'
    execute(client,f'test ! -e {REMOTE} && mkdir -p {CONTROL}')
    started=time.monotonic();sent=0;report=10*1024**2
    stdin,stdout,stderr=client.exec_command('cat > /opt/run064-input.tar.gz',timeout=60)
    with source.open('rb') as handle:
        while data:=handle.read(1024**2):
            stdin.channel.sendall(data);sent+=len(data)
            if sent>=report:
                elapsed=time.monotonic()-started
                print(json.dumps({'stage':'upload','bytes':sent,'total':receipt['bytes'],'seconds':elapsed,'remaining_seconds':(receipt['bytes']-sent)/(sent/elapsed)}),flush=True);report+=10*1024**2
    stdin.channel.shutdown_write();stdout.read();error=stderr.read().decode(errors='replace')
    if stdout.channel.recv_exit_status():raise RuntimeError(error)
    if execute(client,'sha256sum /opt/run064-input.tar.gz').split()[0]!=receipt['sha256']:raise ValueError('Ingress hash mismatch')
    execute(client,f'cp /opt/run064-input.tar.gz /workspace/run064-input.tar.gz && mkdir {REMOTE} && tar -xf /opt/run064-input.tar.gz -C {REMOTE}',timeout=300)
    if execute(client,'sha256sum /workspace/run064-input.tar.gz').split()[0]!=receipt['sha256']:raise ValueError('Persistent hash mismatch')
    command=f'''python3 - <<'PY'
from pathlib import Path
import hashlib,json
root=Path('{REMOTE}')
for row in json.loads((root/'prelaunch/bundle-inventory.json').read_text())['files']:
 p=root/row['path']
 if p.stat().st_size!=row['bytes'] or hashlib.file_digest(p.open('rb'),'sha256').hexdigest()!=row['sha256']:raise ValueError(row['path'])
print('Every bundle member verified')
PY'''
    print(execute(client,command,timeout=300),flush=True)
    with client.open_sftp() as sftp:
        with sftp.open(REMOTE+'/prelaunch/deployment.json','w') as handle:handle.write(json.dumps({'lease':lease,'bundle':receipt,'source_commit':lease['source_commit']}))
    wrapper=f'export RUN064_DEADLINE={lease["deadline_epoch"]}; bash {REMOTE}/04_setup.sh; code=$?; printf "%s\\n" "$code" > {CONTROL}/pipeline.exit'
    execute(client,f'nohup setsid bash -c {shlex.quote(wrapper)} > {CONTROL}/pipeline.log 2>&1 < /dev/null & echo $! > {CONTROL}/pipeline.pid')
    write(ROOT/'prelaunch/upload-001.json',{'upload_seconds':time.monotonic()-started,'bytes':receipt['bytes'],'sha256_verified':True,'members_verified':receipt['files'],'detached':True})
    print('Detached setup/profile pipeline started',flush=True)
def status(client):
    command=f'''python3 - <<'PY'
from pathlib import Path
import json
root=Path('{REMOTE}');control=Path('{CONTROL}')
out={{'pipeline_exit':(control/'pipeline.exit').read_text().strip() if (control/'pipeline.exit').exists() else None,'setup_complete':(root/'runtime/setup-complete').exists(),'log_tail':(control/'pipeline.log').read_text(errors='replace')[-2400:] if (control/'pipeline.log').exists() else ''}}
out['statuses']=[{{'attempt':p.parent.name,**json.loads(p.read_text())}} for p in sorted((root/'artifacts/attempts').glob('*/status.json'))]
precompile=root/'runtime/precompile-status.json'
if precompile.exists():out['precompile']=json.loads(precompile.read_text())
print(json.dumps(out))
PY'''
    print(execute(client,command),flush=True)
    print(execute(client,'nvidia-smi --query-gpu=uuid,name,utilization.gpu,memory.used,memory.total --format=csv,noheader'),flush=True)
def retrieve(client):
    command=f'''python3 - <<'PY'
from pathlib import Path
import hashlib,json,tarfile,shutil
root=Path('{REMOTE}');control=Path('{CONTROL}')
if not (control/'pipeline.exit').exists():raise RuntimeError('Work is active')
(root/'artifacts/control').mkdir(exist_ok=True)
for p in control.iterdir():
 if p.is_file():shutil.copyfile(p,root/'artifacts/control'/p.name)
paths=sorted(p for folder in ['artifacts','runtime'] for p in (root/folder).rglob('*') if p.is_file())
rows=[{{'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()}} for p in paths]
(root/'transfer').mkdir(exist_ok=True)
inventory=root/'transfer/inventory-001.json';inventory.write_text(json.dumps({{'files':rows,'payload_bytes':sum(r['bytes'] for r in rows)}},indent=2)+'\\n')
archive=Path('/opt/run064-output.tar.gz')
with tarfile.open(archive,'w:gz',compresslevel=3) as out:
 for p in paths+[inventory]:out.add(p,arcname=p.relative_to(root).as_posix(),recursive=False)
print(json.dumps({{'bytes':archive.stat().st_size,'sha256':hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()}}))
PY'''
    receipt=json.loads(execute(client,command,timeout=300));target=ROOT/'retrieval/output-001.tar.gz';target.parent.mkdir(exist_ok=True)
    _,out,err=client.exec_command('cat /opt/run064-output.tar.gz',timeout=300)
    with target.open('wb') as handle:
        while data:=out.read(1024**2):handle.write(data)
    if out.channel.recv_exit_status():raise RuntimeError(err.read().decode())
    if target.stat().st_size!=receipt['bytes'] or sha(target)!=receipt['sha256']:raise ValueError('Return archive mismatch')
    with tarfile.open(target) as archive:
        for member in archive:
            path=(ROOT/member.name).resolve()
            if not member.isfile() or not path.is_relative_to(ROOT.resolve()):raise ValueError('Unsafe member')
            path.parent.mkdir(parents=True,exist_ok=True)
            with archive.extractfile(member) as src,path.open('wb') as dst:shutil.copyfileobj(src,dst)
    inventory=read(ROOT/'transfer/inventory-001.json')
    for row in inventory['files']:
        path=ROOT/row['path']
        if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:raise ValueError('Member mismatch '+row['path'])
    write(ROOT/'transfer/receipt-001.json',{'status':'verified','archive':receipt,'verified_files':len(inventory['files']),'payload_bytes':inventory['payload_bytes']})
    print(json.dumps({'verified_files':len(inventory['files']),'payload_bytes':inventory['payload_bytes']}),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','status','retrieve']);args=parser.parse_args()
    client,lease=connect()
    try:globals()[args.action](client,lease) if args.action=='prepare' else globals()[args.action](client)
    finally:client.close()
