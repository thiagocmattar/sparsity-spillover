"""SSH transport for the explicitly recorded Run041 Pod leases."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time
import paramiko

RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
CLI=ROOT/'tmp/runpodctl-v2.14.0.exe'
REMOTE='/workspace/sparsity-spillover'


def connect(role):
    lease=json.loads((RUN/'prelaunch'/f'lease-{role}-001.json').read_text())
    saved=RUN/'prelaunch'/f'ssh-{role}-001.json'
    info=json.loads(saved.read_text()) if saved.exists() else json.loads(subprocess.check_output([str(CLI),'ssh','info',lease['pod']['id']],text=True))
    if info['id']!=lease['pod']['id'] or info['name']!=lease['pod']['name'] or not info['name'].startswith('run041-'):
        raise ValueError('Unscoped Pod')
    saved.write_text(json.dumps(info,indent=2)+'\n')
    known=RUN/'prelaunch/known_hosts'
    known.touch(exist_ok=True)
    client=paramiko.SSHClient();client.load_host_keys(str(known));client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(info['ip'],port=info['port'],username='root',key_filename=info['ssh_key']['path'],
        timeout=20,auth_timeout=20,banner_timeout=20,look_for_keys=False,allow_agent=False)
    client.get_transport().set_keepalive(20)
    return client,lease


def execute(client,command,timeout=60):
    _,out,err=client.exec_command(command,timeout=timeout)
    stdout=out.read().decode(errors='replace');stderr=err.read().decode(errors='replace')
    code=out.channel.recv_exit_status()
    if code:raise RuntimeError(f'Remote exit{code}: {stdout[-3000:]} {stderr[-3000:]}')
    return stdout+stderr


def prepare(client,lease,tag):
    control='/workspace/run041-control'
    execute(client,'mkdir -p '+control)
    with client.open_sftp() as sftp:
        sftp.put(str(RUN/'12_deadline_guard.py'),control+'/12_deadline_guard.py')
        keypath=control+'/guard-settings.json'
        with sftp.open(keypath,'w') as f:
            f.write(json.dumps({'pod_id':lease['pod']['id'],'name':lease['pod']['name'],
                'deadline_epoch':lease['deadline_epoch'],'control':control}))
        sftp.chmod(keypath,0o600)
        execute(client,f'nohup python3 -u {control}/12_deadline_guard.py {keypath} > {control}/guard.log 2>&1 < /dev/null &')
        receipt=json.loads((RUN/'prelaunch'/f'source-{tag}.receipt.json').read_text())
        start=time.monotonic();sftp.put(str(RUN/receipt['path']),'/workspace/run041-source.bundle')
        actual=execute(client,'sha256sum /workspace/run041-source.bundle').split()[0]
        if actual!=receipt['sha256']:raise RuntimeError('Source transfer hash mismatch')
        command=f'test ! -e {REMOTE} && git clone /workspace/run041-source.bundle {REMOTE} && cd {REMOTE} && git checkout '+receipt['deployment_commit']
        execute(client,command)
        if execute(client,f'cd {REMOTE} && git status --porcelain').strip():raise RuntimeError('Dirty deployment')
        excludes=f'{REMOTE}/.git/info/exclude'
        with sftp.open(excludes,'a') as f:
            f.write('\nruns/'+RUN.name+'/prelaunch/\nruns/'+RUN.name+'/artifacts/\n')
        guard=execute(client,f'cat {control}/guard.log; test ! -e {keypath}')
        if '"event": "armed"' not in guard:raise RuntimeError('Workload guard not armed')
        pipeline=f'{REMOTE}/runs/{RUN.name}/08_pipeline.sh'
        wrapper=f'export RUN041_DEADLINE_EPOCH={lease["deadline_epoch"]}; bash {shlex.quote(pipeline)}; code=$?; printf "%s\\n" "$code" > {control}/pipeline.exit'
        result=execute(client,f'nohup setsid bash -c {shlex.quote(wrapper)} > {control}/pipeline.log 2>&1 < /dev/null & echo $! > {control}/pipeline.pid')
        (RUN/'prelaunch/upload-training-001.json').write_text(json.dumps({'source_sha256':actual,'source_bytes':receipt['bytes'],
            'seconds':time.monotonic()-start,'workload_guard':'armed','provider_stop_guard':'local-process','pipeline':'detached'},indent=2)+'\n')
        print('Source verified; guard armed; detached pipeline started',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','command','get']);p.add_argument('--role',default='training')
    p.add_argument('--tag',default='001');p.add_argument('--command-file',type=Path);p.add_argument('--remote');p.add_argument('--local',type=Path)
    a=p.parse_args();client,lease=connect(a.role)
    try:
        if a.action=='prepare':prepare(client,lease,a.tag)
        elif a.action=='command':print(execute(client,a.command_file.read_text(),timeout=60),flush=True)
        else:
            a.local.parent.mkdir(parents=True,exist_ok=True)
            with client.open_sftp() as sftp:sftp.get(a.remote,str(a.local))
            print(json.dumps({'path':str(a.local),'bytes':a.local.stat().st_size}))
    finally:client.close()


if __name__=='__main__':main()
