"""Retrieve only absent inventory files, staging bulk reads on the Pod root disk.

This script does not start Pods, extend guards, or delete resources.
Run only within an approved recovery window with cloud-side guards armed.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time
import paramiko

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('remote032',HERE/'07_remote.py')
remote=importlib.util.module_from_spec(spec)
spec.loader.exec_module(remote)

STAGE_CODE=r'''
import hashlib,json,pathlib,shutil,sys,tarfile,time
settings=json.load(open(sys.argv[1]))
root=pathlib.Path(settings['root'])
stage=pathlib.Path(settings['stage'])
stage.mkdir(exist_ok=False)
for name in settings['files']:
 source=(root/name).resolve()
 if not source.is_relative_to(root/'artifacts') or not source.is_file():raise ValueError(name)
 target=stage/name
 target.parent.mkdir(parents=True,exist_ok=True)
 with source.open('rb') as inp,target.open('xb') as out:shutil.copyfileobj(inp,out,8*1024*1024)
print(json.dumps({'phase':'staged','files':len(settings['files'])}),flush=True)
with tarfile.open(str(stage)+'.tar','w') as archive:
 for name in settings['files']:archive.add(stage/name,arcname=name,recursive=False)
with open(str(stage)+'.tar','rb') as handle:digest=hashlib.file_digest(handle,'sha256').hexdigest()
print(json.dumps({'archive':str(stage)+'.tar','sha256':digest,'bytes':pathlib.Path(str(stage)+'.tar').stat().st_size}),flush=True)
'''

def retrieve(pod,stage_only=False):
    evidence=HERE/'prelaunch/cloud'/pod['id']
    evidence.mkdir(parents=True,exist_ok=True)
    dest=HERE/'retrieval'/pod['id']
    dest.mkdir(parents=True,exist_ok=True)
    root=f'{remote.REMOTE}/{remote.RUN}'
    with remote.connect(pod['id']) as client:
        assert remote.command(client,'cat /workspace/run032-training.exit').strip()=='0'
        with client.open_sftp() as sftp:
            attempts=sftp.listdir(root+'/artifacts/attempts')
            if len(attempts)!=1:raise ValueError('Expected one scientific attempt')
            attempt='artifacts/attempts/'+attempts[0]
            with sftp.open(root+'/'+attempt+'/manifest.json') as handle:manifest=json.load(handle)
            assert manifest['condition']['id']==pod['condition'] and manifest['status']=='completed'
            with sftp.open(root+'/'+attempt+'/transfer_inventory.json') as handle:inventory=json.load(handle)
            files=[]
            for entry in inventory['files']:
                name=attempt+'/'+entry['path']
                local=HERE/name
                if local.exists():
                    if local.stat().st_size!=entry['bytes'] or remote.sha(local)!=entry['sha256']:raise ValueError('Existing artifact differs: '+name)
                else:files.append(name)
            for name in [attempt+'/transfer_inventory.json',f"artifacts/workers/{pod['condition']}/progress.json"]:
                if not (HERE/name).exists():files.append(name)
            for name in sftp.listdir('/workspace'):
                if name.startswith('run032-') and name.endswith(('.log','.exit','.sha256')):
                    sftp.get('/workspace/'+name,str(evidence/name))
            stage='/opt/run032-recovery-'+str(int(time.time()))
            settings={'root':root,'stage':stage,'files':files}
            with sftp.open('/tmp/run032-stage-settings.json','w') as handle:handle.write(json.dumps(settings))
            with sftp.open('/tmp/run032-stage.py','w') as handle:handle.write(STAGE_CODE)
        print(json.dumps({'pod':pod['id'],'phase':'staging_missing_files','file_count':len(files)}),flush=True)
        output=remote.command(client,'python3 /tmp/run032-stage.py /tmp/run032-stage-settings.json')
        package=json.loads(output.strip().splitlines()[-1])
        if stage_only:
            (evidence/'staged-package.json').write_text(json.dumps(package,indent=2)+'\n',encoding='utf-8')
            print(json.dumps({'pod':pod['id'],'phase':'staged','package':package}),flush=True)
            return
        archive_path=dest/('missing-'+str(int(time.time()))+'.tar')
        with paramiko.SFTPClient.from_transport(client.get_transport(),window_size=32*1024**2) as sftp:
            sftp.get(package['archive'],str(archive_path),prefetch=True,max_concurrent_prefetch_requests=64)
    assert archive_path.stat().st_size==package['bytes'] and remote.sha(archive_path)==package['sha256']
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            target=(HERE/member.name).resolve()
            if not member.isfile() or not target.is_relative_to((HERE/'artifacts').resolve()):raise ValueError('Unsafe member')
            if target.exists():raise FileExistsError(target)
        archive.extractall(HERE,filter='data')
    subprocess.run([sys.executable,str(HERE/'03_verify.py'),'--condition',pod['condition']],check=True)
    receipt={'pod':pod['id'],'condition':pod['condition'],'retrieved_utc':datetime.now(timezone.utc).isoformat(),'standalone_verification':'passed','method':'Original per-file inventory verified after staged missing-file retrieval; earlier salvage provenance retained','package':package}
    (evidence/'retrieval-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--pod',required=True)
    parser.add_argument('--stage-only',action='store_true')
    args=parser.parse_args()
    pods=json.loads((HERE/'prelaunch/pods.json').read_text())
    retrieve(next(p for p in pods if p['id']==args.pod),args.stage_only)
