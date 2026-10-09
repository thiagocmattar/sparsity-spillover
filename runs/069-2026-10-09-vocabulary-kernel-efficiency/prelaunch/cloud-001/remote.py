"""Run069 attempt 001 transport; provider credentials stay on the workstation."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time
import paramiko

ATTEMPT=Path(__file__).resolve().parent
RUN=ATTEMPT.parents[1]
REPO=RUN.parents[1]
REMOTE='/workspace/run069'
RELATIVE=RUN.relative_to(REPO).as_posix()
CONTROL='/workspace/run069-control'


def execute(client,command,timeout=60):
    _,out,err=client.exec_command(command,timeout=timeout)
    stdout=out.read().decode(errors='replace');stderr=err.read().decode(errors='replace')
    code=out.channel.recv_exit_status()
    if code:raise RuntimeError(f'Remote exit {code}: {stdout[-3000:]} {stderr[-3000:]}')
    return stdout+stderr


def connect():
    lease=json.loads((ATTEMPT/'lease.json').read_text())
    info=json.loads(subprocess.check_output([str(REPO/'tmp/runpodctl-v2.14.0.exe'),'ssh','info',lease['pod']['id'],'--verbose'],text=True))
    if info['id']!=lease['pod']['id'] or info['name']!='run069-vocabulary-efficiency-001':
        raise ValueError('Unscoped SSH target')
    known=ATTEMPT/'known_hosts';known.touch(exist_ok=True)
    client=paramiko.SSHClient();client.load_host_keys(str(known));client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(info['ip'],port=info['port'],username='root',key_filename=info['ssh_key']['path'],
                   timeout=20,auth_timeout=20,banner_timeout=20,look_for_keys=False,allow_agent=False)
    client.get_transport().set_keepalive(20)
    return client,lease


def prepare(client,lease):
    execute(client,f'test ! -e {CONTROL}/pipeline.pid && mkdir -p {REMOTE} {CONTROL}')
    receipt=json.loads((RUN/'prelaunch/bundle.json').read_text())['archive']
    archive=REPO/receipt['path'];remote='/workspace/run069-upload.tar.gz'
    transport=client.get_transport()
    with paramiko.SFTPClient.from_transport(transport,window_size=64*1024**2,max_packet_size=1024**2) as sftp:
        started=time.monotonic();last=started
        def progress(done,total):
            nonlocal last
            now=time.monotonic()
            if now-last>=30 or done==total:
                print(json.dumps({'stage':'upload','bytes':done,'total':total,'MiB_per_second':done/1024**2/max(.01,now-started)}),flush=True);last=now
        sftp.put(str(archive),remote,callback=progress)
        if execute(client,'sha256sum '+remote).split()[0]!=receipt['sha256']:raise ValueError('Archive transfer hash mismatch')
        execute(client,f'tar -xzf {remote} -C {REMOTE}',timeout=120)
        code=f"import sys;sys.path.insert(0,{RELATIVE!r});from common import manifest_identity;print(manifest_identity())"
        check=execute(client,f'cd {REMOTE} && RUN069_SOURCE_COMMIT={lease["source_commit"]} python3 -c '+shlex.quote(code),timeout=120)
        (ATTEMPT/'remote-source-check.txt').write_text(check,encoding='utf-8')
        wrapper=(f'#!/usr/bin/env bash\nset -euo pipefail\n'
                 f'export RUN069_SOURCE_COMMIT={lease["source_commit"]}\n'
                 f'trap \'code=$?; printf "%s\\n" "$code" > {CONTROL}/pipeline.exit\' EXIT\n'
                 f'bash {REMOTE}/{RELATIVE}/00_setup.sh\n'
                 f'bash {REMOTE}/{RELATIVE}/07_execute.sh\n')
        with sftp.open(CONTROL+'/launch.sh','w') as stream:stream.write(wrapper)
    execute(client,f'nohup setsid bash {CONTROL}/launch.sh > {CONTROL}/pipeline.log 2>&1 < /dev/null & echo $! > {CONTROL}/pipeline.pid')
    print(json.dumps({'stage':'detached','pod_id':lease['pod']['id'],'source_commit':lease['source_commit']}),flush=True)


def status(client):
    code=f'''import json,os,time
from pathlib import Path
control=Path({CONTROL!r});root=Path({REMOTE+'/'+RELATIVE!r})
pid=int((control/'pipeline.pid').read_text()) if (control/'pipeline.pid').exists() else None
live=pid is not None and Path('/proc/'+str(pid)).exists()
events=[]
for path in root.glob('artifacts/**/status.json'):
    events.append(json.loads(path.read_text()))
events.sort(key=lambda x:x['utc'])
log=control/'pipeline.log'
with log.open('rb') as stream:
    stream.seek(max(0,log.stat().st_size-8000));tail=stream.read().decode(errors='replace')
print(json.dumps({{'pid':pid,'live':live,'exit_code':(control/'pipeline.exit').read_text().strip() if (control/'pipeline.exit').exists() else None,'recent_events':events[-8:],'log_tail':tail,'disk_free_bytes':os.statvfs('/workspace').f_bavail*os.statvfs('/workspace').f_frsize}}))
'''
    output=execute(client,'python3 -c '+shlex.quote(code));print(output,flush=True)
    (ATTEMPT/'latest-status.json').write_text(output,encoding='utf-8')


def retrieve(client):
    if execute(client,f'cat {CONTROL}/pipeline.exit').strip()!='0':raise ValueError('Pipeline did not complete')
    execute(client,f'cp {CONTROL}/pipeline.log {REMOTE}/{RELATIVE}/artifacts/control/cloud-pipeline.txt')
    print(execute(client,f'cd {REMOTE} && RUN069_SOURCE_COMMIT='+json.loads((ATTEMPT/'lease.json').read_text())['source_commit']+f' /workspace/run069-env/bin/python {RELATIVE}/08_verify.py',timeout=120),flush=True)
    with paramiko.SFTPClient.from_transport(client.get_transport(),window_size=64*1024**2,max_packet_size=1024**2) as sftp:
        with sftp.open(f'{REMOTE}/{RELATIVE}/transfer/transfer-inventory.json') as stream:data=stream.read()
        rows=json.loads(data);total=sum(r['bytes'] for r in rows);done=0;started=time.monotonic();last=started
        for row in rows:
            local=(REPO/row['path']).resolve()
            if not local.is_relative_to(RUN) or local.suffix=='.tmp':raise ValueError('Unscoped artifact')
            local.parent.mkdir(parents=True,exist_ok=True);temporary=local.with_suffix(local.suffix+'.tmp')
            sftp.get(REMOTE+'/'+row['path'],str(temporary),max_concurrent_prefetch_requests=16)
            with temporary.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
            if temporary.stat().st_size!=row['bytes'] or digest!=row['sha256']:raise ValueError('Artifact hash mismatch: '+row['path'])
            if local.exists() and local.read_bytes()!=temporary.read_bytes():raise ValueError('Refusing to overwrite different retained evidence')
            temporary.replace(local);done+=row['bytes'];now=time.monotonic()
            if now-last>=30 or done==total:
                print(json.dumps({'stage':'retrieve','bytes':done,'total':total,'MiB_per_second':done/1024**2/max(.01,now-started)}),flush=True);last=now
        dest=RUN/'transfer/transfer-inventory.json';dest.parent.mkdir(exist_ok=True);dest.write_bytes(data)
        (ATTEMPT/'retrieval.json').write_text(json.dumps({'status':'verified','files':len(rows),'bytes':total,'inventory_sha256':hashlib.sha256(data).hexdigest()},indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','status','retrieve']);args=p.parse_args()
    client,lease=connect()
    try:{'prepare':lambda:prepare(client,lease),'status':lambda:status(client),'retrieve':lambda:retrieve(client)}[args.action]()
    finally:client.close()
