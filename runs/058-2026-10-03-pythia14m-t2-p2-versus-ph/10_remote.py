"""Scoped SSH transport; API credentials remain on the workstation."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import socket
import subprocess
import time
import paramiko
from run_config import RUN_DIR as RUN,REPO_ROOT as ROOT,write_json

CLI=ROOT/'tmp/runpodctl-v2.14.0.exe'
REMOTE='/workspace/sparsity-spillover'

def connect(role='training'):
    lease=json.loads((RUN/'prelaunch'/f'lease-{role}-001.json').read_text())
    info=json.loads(subprocess.check_output([str(CLI),'ssh','info',lease['pod']['id']],text=True))
    if info['id']!=lease['pod']['id'] or info['name']!=lease['pod']['name'] or not info['name'].startswith('run058-'):
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

def upload(sftp,local,remote):
    local=Path(local);total=local.stat().st_size
    try:offset=sftp.stat(remote).st_size
    except FileNotFoundError:offset=0
    if not 0<=offset<=total:raise ValueError('Remote size exceeds immutable source')
    started=time.monotonic();last=started;done=offset
    with local.open('rb') as source,sftp.open(remote,'ab' if offset else 'wb') as target:
        target.set_pipelined(True);source.seek(offset)
        while chunk:=source.read(1024**2):
            target.write(chunk);done+=len(chunk)
            now=time.monotonic()
            if now-last>=30 or done==total:
                print(json.dumps({'event':'upload','file':local.name,'bytes':done,'total':total,
                    'MiB_per_second':(done-offset)/1024**2/max(.01,now-started)}),flush=True)
                last=now

def prepare(client,lease,tag):
    control='/workspace/run058-control'
    execute(client,f'mkdir -p {control}')
    transport=client.get_transport();transport.sock.setsockopt(socket.SOL_SOCKET,socket.SO_SNDBUF,8*1024**2)
    with paramiko.SFTPClient.from_transport(transport,window_size=64*1024**2,max_packet_size=1024**2) as sftp:
        sftp.put(str(RUN/'12_deadline_guard.py'),control+'/12_deadline_guard.py')
        settings=control+'/guard-settings.json'
        with sftp.open(settings,'w') as handle:
            handle.write(json.dumps({'pod_id':lease['pod']['id'],'name':lease['pod']['name'],
                'deadline_epoch':lease['deadline_epoch'],'control':control}))
        sftp.chmod(settings,0o600)
        execute(client,f'nohup python3 -u {control}/12_deadline_guard.py {settings} > {control}/guard.log 2>&1 < /dev/null &')
        receipt=json.loads((RUN/'prelaunch'/f'source-{tag}.receipt.json').read_text())
        upload(sftp,RUN/receipt['path'],'/workspace/run058-source.bundle')
        if execute(client,'sha256sum /workspace/run058-source.bundle').split()[0]!=receipt['sha256']:
            raise RuntimeError('Source transfer hash mismatch')
        execute(client,f'test ! -e {REMOTE} && git clone --no-checkout /workspace/run058-source.bundle {REMOTE}',timeout=120)
        execute(client,f'cd {REMOTE} && git sparse-checkout init --cone && git sparse-checkout set src runs/004-2026-08-29-pythia14m-full-pass-l1n runs/{RUN.name} && git checkout '+receipt['deployment_commit'],timeout=120)
        if execute(client,f'cd {REMOTE} && git rev-parse HEAD').strip()!=receipt['deployment_commit']:
            raise RuntimeError('Remote Git commit differs')
        run=f'{REMOTE}/runs/{RUN.name}'
        replay='inputs/random-initialization/model.safetensors'
        upload(sftp,RUN/replay,f'{run}/{replay}')
        provenance=json.loads((RUN/'inputs/random-initialization/provenance.json').read_text())
        if execute(client,f'sha256sum {run}/{replay}').split()[0]!=provenance['file_sha256']:
            raise RuntimeError('Random replay transfer differs')
        with sftp.open(f'{REMOTE}/.git/info/exclude','a') as handle:
            handle.write(f'\nruns/{RUN.name}/prelaunch/\nruns/{RUN.name}/artifacts/\nruns/{RUN.name}/transfer/\n')
        wrapper=(f'cd {REMOTE}; bash {run}/00_setup_remote.sh || exit $?; '
            f'/workspace/run058-venv/bin/python {run}/06_build_cache_from_hf.py > {control}/cache-build.log 2>&1 || exit $?; '
            f'touch {control}/cache-ready; '
            f'bash {run}/08_pipeline.sh; code=$?; printf "%s\\n" "$code" > {control}/pipeline.exit')
        execute(client,f'nohup setsid bash -c {shlex.quote(wrapper)} > {control}/pipeline.log 2>&1 < /dev/null & echo $! > {control}/pipeline.pid')
    write_json(RUN/'prelaunch/upload-training-001.json',{'source':receipt,'cache_verification':'remote hash-locked rebuild before preflight',
        'workload_guard':'credential-free remote process','provider_guard':'workstation stop process',
        'pipeline':'detached; frozen pinned HF cache builder; fresh preflight barrier'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','command','get'])
    p.add_argument('--tag',default='001');p.add_argument('--command-file',type=Path)
    p.add_argument('--remote');p.add_argument('--local',type=Path)
    args=p.parse_args();client,lease=connect()
    try:
        if args.action=='prepare':prepare(client,lease,args.tag)
        elif args.action=='command':print(execute(client,args.command_file.read_text()),flush=True)
        else:
            args.local.parent.mkdir(parents=True,exist_ok=True)
            with client.open_sftp() as sftp:sftp.get(args.remote,str(args.local))
    finally:client.close()
