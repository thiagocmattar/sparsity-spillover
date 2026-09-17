"""Run034 scoped provisioning/SSH/input transport. No scientific scheduling."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time
import tomllib
import paramiko

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CLI = ROOT/'tmp/runpodctl-v2.12.0.exe'
REMOTE = '/workspace/sparsity-spillover'
REMOTE_RUN = REMOTE+'/runs/'+HERE.name
CONTROL = '/workspace/run034-control'


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n')


def cli(*args):
    return json.loads(subprocess.check_output([str(CLI),*args],text=True))


def create(label, counts):
    import yaml
    cfg=yaml.safe_load((HERE/'config.yaml').read_text())['runpod']
    catalog=cli('gpu','list','--include-unavailable')
    before=cli('pod','list','--all')
    active=[p for p in before if p.get('name','').startswith('run034-')]
    used=sum(int(p.get('gpuCount',1)) for p in active)
    assert used+min(counts)<=10, 'GPU envelope would be exceeded'
    attempt_file=HERE/'prelaunch'/f'allocations-{label}.json'
    attempts=[]
    for gpu in cfg['candidate_gpu_types']:
        for count in counts:
            if used+count>10:continue
            for tier in ('COMMUNITY','SECURE'):
                name=f'run034-{label}'
                started=time.time()
                cmd=['pod','create','--name',name,'--image',cfg['image'],'--gpu-id',gpu,
                     '--gpu-count',str(count),'--cloud-type',tier,'--container-disk-in-gb','40',
                     '--volume-in-gb',str(20+20*count),'--ports','22/tcp','--ssh',
                     '--min-cuda-version','12.8']
                if tier=='COMMUNITY':cmd.append('--public-ip')
                result=subprocess.run([str(CLI),*cmd],capture_output=True,text=True)
                if result.returncode:
                    # Check provider state before retrying an ambiguous create response.
                    matches=[p for p in cli('pod','list','--all') if p.get('name')==name]
                    attempts.append({'gpu':gpu,'count':count,'tier':tier,'error':result.stderr[-2000:]+result.stdout[-2000:]})
                    write(attempt_file,attempts)
                    if matches:raise RuntimeError('Creation response failed but Pod exists; reconcile before retry')
                    continue
                pod=json.loads(result.stdout)
                price=next(r for r in catalog if r['gpuId']==gpu)[tier.lower()+'PricePerHr']
                lease={'pod':pod,'label':label,'gpu_type':gpu,'gpu_count':count,'tier':tier,
                       'quoted_per_gpu_hour_usd':price,'created_request_epoch':started,
                       'deadline_epoch':started+4*3600,'command_without_secrets':cmd,
                       'captured_at':datetime.now(timezone.utc).isoformat()}
                write(HERE/'prelaunch'/f'lease-{label}.json',lease)
                attempts.append({'gpu':gpu,'count':count,'tier':tier,'pod_id':pod['id'],'status':'created'})
                write(attempt_file,attempts)
                print(json.dumps({'pod':pod['id'],'name':name,'gpu':gpu,'count':count,'tier':tier,'price_per_gpu_hour':price}),flush=True)
                return
    raise RuntimeError('No capacity found for bounded candidate set')


def connect(label,refresh=False):
    lease=json.loads((HERE/'prelaunch'/f'lease-{label}.json').read_text())
    saved=HERE/'prelaunch'/f'ssh-{label}.json'
    info=cli('ssh','info',lease['pod']['id']) if refresh or not saved.exists() else json.loads(saved.read_text())
    assert info['id']==lease['pod']['id'] and info['name']=='run034-'+label
    if not info.get('ip') or not info.get('port'):raise RuntimeError('Public SSH not ready')
    write(saved,info)
    client=paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(info['ip'],port=info['port'],username='root',key_filename=info['ssh_key']['path'],
                   timeout=20,banner_timeout=20,auth_timeout=20,look_for_keys=False,allow_agent=False)
    client.get_transport().set_keepalive(20)
    return client


def command(client,text,timeout=60):
    _,out,err=client.exec_command(text,timeout=timeout)
    stdout=out.read().decode(errors='replace');stderr=err.read().decode(errors='replace')
    code=out.channel.recv_exit_status()
    if code:raise RuntimeError(f'Remote exit {code}: {stdout[-2000:]} {stderr[-2000:]}')
    return stdout


def upload(client,local,remote):
    command(client,'mkdir -p '+shlex.quote(remote.rsplit('/',1)[0]))
    started=time.time();last=[started]
    def progress(done,total):
        now=time.time()
        if now-last[0]>30 or done==total:
            print(json.dumps({'file':str(local),'bytes':done,'total':total,'MB_s':done/1e6/max(now-started,.001)}),flush=True);last[0]=now
    with client.open_sftp() as sftp:sftp.put(str(local),remote,callback=progress)


def arm(label):
    lease=json.loads((HERE/'prelaunch'/f'lease-{label}.json').read_text())
    with connect(label) as client:
        command(client,'mkdir -p '+CONTROL)
        upload(client,HERE/'08_deadline_guard.py',CONTROL+'/deadline_guard.py')
        key=tomllib.loads((Path.home()/'.runpod/config.toml').read_text())['apikey']
        settings={'api_key':key,'pod_id':lease['pod']['id'],'name':'run034-'+label,'deadline_epoch':lease['deadline_epoch']}
        with client.open_sftp() as sftp:
            with sftp.open('/tmp/run034-stop-secret','w') as h:h.write(json.dumps(settings))
            sftp.chmod('/tmp/run034-stop-secret',0o600)
        command(client,f'setsid python3 {CONTROL}/deadline_guard.py /tmp/run034-stop-secret > {CONTROL}/guard.log 2>&1 < /dev/null & echo $! > {CONTROL}/guard.pid')
    print('On-Pod guard started for '+label)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['create','arm','command','upload']);p.add_argument('--label',required=True);p.add_argument('--counts',default='4,2,1');p.add_argument('--file',type=Path);p.add_argument('--remote')
    a=p.parse_args()
    if a.action=='create':create(a.label,[int(v) for v in a.counts.split(',')])
    elif a.action=='arm':arm(a.label)
    else:
        with connect(a.label) as c:
            if a.action=='command':print(command(c,a.file.read_text(encoding='utf-8')))
            else:upload(c,a.file,a.remote)
