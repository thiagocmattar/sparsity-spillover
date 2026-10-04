"""Owned-Pod SSH setup and hash-verified artifact retrieval."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import socket
import subprocess
import tarfile
import time
import paramiko
from io_utils import RUN,REPO,read,write,sha,record

CLI=REPO/'tmp/runpodctl-v2.14.0.exe'
REMOTE='/workspace/run059'
CONTROL='/workspace/run059-control'


def connect():
    lease=read(RUN/'prelaunch/lease-001.json')
    info=json.loads(subprocess.check_output([str(CLI),'ssh','info',lease['pod']['id']],text=True))
    if info['id']!=lease['pod']['id'] or info['name']!=lease['pod']['name'] or not info['name'].startswith('run059-'):
        raise ValueError('Unscoped SSH target')
    known=RUN/'prelaunch/known_hosts';known.touch(exist_ok=True)
    client=paramiko.SSHClient();client.load_host_keys(str(known));client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(info['ip'],port=info['port'],username='root',key_filename=info['ssh_key']['path'],
        timeout=20,auth_timeout=20,banner_timeout=20,look_for_keys=False,allow_agent=False)
    client.get_transport().set_keepalive(20)
    return client,lease


def execute(client,command,timeout=60):
    _,out,err=client.exec_command(command,timeout=timeout)
    stdout=out.read().decode(errors='replace');stderr=err.read().decode(errors='replace')
    code=out.channel.recv_exit_status()
    if code:raise RuntimeError(f'Remote exit {code}: {stdout[-3000:]} {stderr[-3000:]}')
    return stdout+stderr


def prepare(client,lease):
    receipt=read(RUN/'bundles/input-001.receipt.json')
    transport=client.get_transport();transport.sock.setsockopt(socket.SOL_SOCKET,socket.SO_SNDBUF,8*1024**2)
    execute(client,f'mkdir -p {CONTROL}')
    with paramiko.SFTPClient.from_transport(transport,window_size=64*1024**2,max_packet_size=1024**2) as sftp:
        sftp.put(str(RUN/'08_deadline_guard.py'),CONTROL+'/guard.py')
        with sftp.open(CONTROL+'/guard-settings.json','w') as handle:
            handle.write(json.dumps({'pod_id':lease['pod']['id'],'name':lease['pod']['name'],
                'deadline_epoch':lease['deadline_epoch'],'control':CONTROL}))
        execute(client,f'nohup python3 -u {CONTROL}/guard.py {CONTROL}/guard-settings.json > {CONTROL}/guard.log 2>&1 < /dev/null &')
        started=time.monotonic()
        sftp.put(str(RUN/receipt['path']),'/workspace/run059-input.tar')
        upload_seconds=time.monotonic()-started
        if execute(client,'sha256sum /workspace/run059-input.tar').split()[0]!=receipt['sha256']:raise RuntimeError('Input archive transfer hash mismatch')
        execute(client,f'test ! -e {REMOTE} && mkdir {REMOTE} && tar -xf /workspace/run059-input.tar -C {REMOTE}',timeout=120)
        wrapper=(f'cd {REMOTE}; bash {REMOTE}/04_setup.sh || exit $?; '
            f'bash {REMOTE}/05_execute.sh smoke {lease["deadline_epoch"]} || exit $?; '
            f'bash {REMOTE}/05_execute.sh scientific {lease["deadline_epoch"]}')
        wrapper=f'({wrapper}); code=$?; printf "%s\\n" "$code" > {CONTROL}/pipeline.exit'
        execute(client,f'nohup setsid bash -c {shlex.quote(wrapper)} > {CONTROL}/pipeline.log 2>&1 < /dev/null & echo $! > {CONTROL}/pipeline.pid')
    write(RUN/'prelaunch/upload-001.json',{'source':receipt,'upload_seconds':upload_seconds,
        'archive_sha256_verified':True,'detached':True,'remote_guard':'workload-only','provider_guard':'independent hidden workstation guard'})
    print(json.dumps({'uploaded_bytes':receipt['bytes'],'upload_seconds':upload_seconds,'launched':True}))


def status(client):
    command=f"""python3 - <<'PY'
from pathlib import Path
import json
run=Path('{REMOTE}');control=Path('{CONTROL}')
output={{'pipeline_exit':(control/'pipeline.exit').read_text().strip() if (control/'pipeline.exit').exists() else None,
    'setup_complete':(run/'runtime/setup-complete').exists(),'statuses':[],
    'pipeline_tail':(control/'pipeline.log').read_text(errors='replace')[-1800:]}}
for path in sorted((run/'artifacts/attempts').glob('*/status.json')):
    output['statuses'].append({{'attempt':path.parent.name,**json.loads(path.read_text())}})
print(json.dumps(output))
PY"""
    print(execute(client,command),flush=True)
    print(execute(client,'nvidia-smi --query-gpu=uuid,name,utilization.gpu,memory.used,memory.total --format=csv,noheader'),flush=True)


def retrieve(client):
    command=f"""python3 - <<'PY'
from pathlib import Path
import hashlib,json,shutil,tarfile
run=Path('{REMOTE}');control=Path('{CONTROL}')
if not (control/'pipeline.exit').exists():raise RuntimeError('Pipeline is still active')
out=run/'artifacts/control';out.mkdir(exist_ok=True)
for name in ['pipeline.log','pipeline.exit','pipeline.pid','guard.log']:
    if (control/name).exists():shutil.copyfile(control/name,out/name)
paths=sorted(p for p in (run/'artifacts').rglob('*') if p.is_file())
paths += [run/'runtime'/name for name in ['pip-freeze.txt','nvcc.txt','nvidia-smi-initial.txt']]
rows=[{{'path':p.relative_to(run).as_posix(),'bytes':p.stat().st_size,
    'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()}} for p in paths]
inventory={{'files':rows,'payload_bytes':sum(r['bytes'] for r in rows)}}
(run/'transfer').mkdir(exist_ok=True)
(run/'transfer/inventory-001.json').write_text(json.dumps(inventory,indent=2)+'\\n')
archive=Path('/workspace/run059-output.tar.gz')
with tarfile.open(archive,'w:gz') as tar:
    for p in paths+[run/'transfer/inventory-001.json']:tar.add(p,arcname=p.relative_to(run).as_posix(),recursive=False)
receipt={{'bytes':archive.stat().st_size,'sha256':hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()}}
print(json.dumps(receipt))
PY"""
    receipt=json.loads(execute(client,command,timeout=120))
    target=RUN/'retrieval/output-001.tar.gz';target.parent.mkdir(exist_ok=True)
    with client.open_sftp() as sftp:sftp.get('/workspace/run059-output.tar.gz',str(target))
    if target.stat().st_size!=receipt['bytes'] or sha(target)!=receipt['sha256']:raise RuntimeError('Returned archive hash mismatch')
    with tarfile.open(target) as archive:
        for member in archive:
            path=(RUN/member.name).resolve()
            if not path.is_relative_to(RUN) or not member.isfile():raise RuntimeError('Unsafe archive member')
            path.parent.mkdir(parents=True,exist_ok=True)
            with archive.extractfile(member) as source,path.open('wb') as destination:
                __import__('shutil').copyfileobj(source,destination)
    inventory=read(RUN/'transfer/inventory-001.json')
    for row in inventory['files']:
        if record(RUN/row['path'])!=row:raise RuntimeError('Returned member hash mismatch')
    write(RUN/'transfer/receipt-001.json',{'archive':receipt,'verified_files':len(inventory['files']),
        'payload_bytes':inventory['payload_bytes'],'status':'verified'})
    print(json.dumps({'verified_files':len(inventory['files']),'payload_bytes':inventory['payload_bytes']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','status','retrieve'])
    args=parser.parse_args();client,lease=connect()
    try:
        if args.action=='prepare':prepare(client,lease)
        elif args.action=='status':status(client)
        else:retrieve(client)
    finally:client.close()
