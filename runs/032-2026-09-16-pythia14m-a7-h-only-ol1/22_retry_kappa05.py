"""One bounded recovery attempt for the last existing Pod; no retraining.

The PowerShell caller waits five minutes between exit-code-75 capacity misses.
--prepare installs a boot-time one-hour stop guard before any resume attempt.
Credentials are read from the existing CLI config and never logged or saved
locally. The guard receives its key through the Pod environment; the service
startup unsets that variable, and deleting the recovered Pod removes its config.
"""
import argparse
import base64
from datetime import datetime,timezone
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.request

HERE=Path(__file__).resolve().parent
POD='8o1uyxgh6vb6ym'
NAME='run032-k0p5'
KEY=tomllib.loads((Path.home()/'.runpod/config.toml').read_text())['apikey']
spec=importlib.util.spec_from_file_location('remote032',HERE/'07_remote.py')
remote=importlib.util.module_from_spec(spec)
spec.loader.exec_module(remote)
BOOT_GUARD='''import os,time,json,urllib.request,urllib.error
pod=os.environ['RUNPOD_POD_ID']
key=os.environ.pop('RUN032_RECOVERY_STOP_KEY')
deadline=time.time()+3600
url='https://api.runpod.io/v2/pods/'+pod
headers={'Authorization':'Bearer '+key,'User-Agent':'run032-recovery','Content-Type':'application/json'}
def request(stop=False):
 r=urllib.request.Request(url+'/action' if stop else url,data=b'{"action":"stop"}' if stop else None,headers=headers,method='POST' if stop else 'GET')
 with urllib.request.urlopen(r,timeout=20) as f:return json.load(f)
while True:
 try:
  current=request()
  assert current['id']==pod and current['name'].startswith('run032-k0p5')
  break
 except Exception as e:
  print(type(e).__name__,getattr(e,'code',None),flush=True)
  time.sleep(10)
print(json.dumps({'event':'armed','pod':pod,'deadline_epoch':deadline}),flush=True)
while time.time()<deadline:time.sleep(min(30,max(0,deadline-time.time())))
while True:
 try:
  request(True)
  print('STOP_REQUESTED',flush=True)
  break
 except Exception as e:
  print(type(e).__name__,getattr(e,'code',None),flush=True)
  if getattr(e,'code',None)==404:break
  time.sleep(10)
'''

def request(path,body=None,method='GET',graphql=False):
    url='https://api.runpod.io/graphql' if graphql else 'https://api.runpod.io/v2'+path
    req=urllib.request.Request(url,data=None if body is None else json.dumps(body).encode(),headers={'Authorization':'Bearer '+KEY,'User-Agent':'run032-recovery','Content-Type':'application/json'},method=method)
    with urllib.request.urlopen(req,timeout=30) as response:
        data=response.read()
        return json.loads(data) if data else {}

def emit(event,**fields):
    row={'utc':datetime.now(timezone.utc).isoformat(),'event':event,**fields}
    print(json.dumps(row),flush=True)
    with (HERE/'prelaunch/kappa05-persistent-recovery.jsonl').open('a',encoding='utf-8') as handle:handle.write(json.dumps(row)+'\n')

def main(prepare):
    pod=request('/pods/'+POD)
    assert pod['id']==POD and pod['name']==NAME
    if prepare:
        assert pod['status']=='EXITED'
        payload=base64.b64encode(BOOT_GUARD.encode()).decode()
        command='python3 -c '+shlex.quote('import base64;exec(base64.b64decode('+repr(payload)+'))')+' > /workspace/run032-boot-recovery-guard.log 2>&1 < /dev/null & unset RUN032_RECOVERY_STOP_KEY; exec /start.sh'
        env={**pod.get('env',{}),'RUN032_RECOVERY_STOP_KEY':KEY}
        request('/pods/'+POD,{'args':'/bin/bash -lc '+shlex.quote(command),'env':env},'PATCH')
        emit('boot_guard_prepared',pod=POD,guard_seconds=3600)
        return 0
    if pod['status']=='EXITED':
        result=request('',{'query':'mutation { podResume(input: { podId: "'+POD+'", gpuCount: 0, computeType: CPU }) { id desiredStatus gpuCount costPerHr } }'},'POST',graphql=True)
        if result.get('errors'):
            try:pod=request('/pods/'+POD+'/action',{'action':'start'},'POST')
            except urllib.error.HTTPError as error:
                if error.code not in (400,409):raise
                details=json.loads(error.read())
                emit('capacity_unavailable',pod=POD,cpu_error=result['errors'][0]['message'],gpu_error=details.get('detail'))
                return 75
        else:emit('cpu_resume_accepted',result=result['data']['podResume'])
    for _ in range(24):
        pod=request('/pods/'+POD)
        if pod['status']=='EXITED':return 75
        direct=(pod.get('ssh') or {}).get('direct')
        if direct:
            values=json.loads((HERE/'prelaunch/recovery-ssh.json').read_text())
            values=[v for v in values if v['id']!=POD]+[{'id':POD,'name':NAME,'status':pod['status'],'ip':direct['host'],'port':direct['port']}]
            (HERE/'prelaunch/recovery-ssh.json').write_text(json.dumps(values,indent=2)+'\n')
            try:
                with remote.connect(POD) as client:
                    guard=remote.command(client,'cat /workspace/run032-boot-recovery-guard.log')
                    if '"event": "armed"' not in guard:raise RuntimeError('Guard not yet armed')
                    (HERE/'prelaunch/cloud'/POD/'boot-guard-armed.json').write_text(guard)
                    state=remote.command(client,'cat /workspace/run032-training.exit; ls -l /workspace/run032-results.tar')
                    emit('access_restored',pod=POD,training_state=state)
                break
            except (OSError,RuntimeError):pass
        time.sleep(10)
    else:
        request('/pods/'+POD+'/action',{'action':'stop'},'POST')
        emit('ssh_not_ready_stopped',pod=POD)
        return 75
    subprocess.run([sys.executable,str(HERE/'19_retrieve_missing.py'),'--pod',POD,'--stage-only'],check=True)
    subprocess.run([sys.executable,str(HERE/'21_stream_staged_archive.py'),'--pod',POD],check=True)
    receipt=json.loads((HERE/'prelaunch/cloud'/POD/'retrieval-receipt.json').read_text())
    assert receipt['pod']==POD and receipt['standalone_verification']=='passed'
    pod=request('/pods/'+POD)
    assert pod['id']==POD and pod['name']==NAME
    request('/pods/'+POD,method='DELETE')
    emit('recovery_verified_pod_deleted',pod=POD)
    (HERE/'prelaunch/kappa05-recovery-complete.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return 0

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    sys.exit(main(args.prepare))
