"""Retrieve one terminal training Pod; retain and check every byte before teardown."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone

RUN = Path(__file__).resolve().parent.parent
REMOTE = '/workspace/sparsity-spillover/runs/'+RUN.name
KEY = str(Path.home()/'.runpod/ssh/runpodctl-ssh-key')


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--connections',default='parallel-connections-001.json')
    p.add_argument('--node',type=int,required=True)
    p.add_argument('--tag',required=True)
    p.add_argument('--retired',action='store_true')
    p.add_argument('--reuse-completed-seed',action='store_true')
    args=p.parse_args()
    if not args.tag.replace('-','').isalnum():raise ValueError('Invalid tag')
    node=json.loads((RUN/'prelaunch'/args.connections).read_text())[args.node]
    s=node['ssh'];host=s['username']+'@'+s['host']
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-i',KEY,'-p',str(s['port']),host]
    scp=['scp','-q','-o','BatchMode=yes','-i',KEY,'-P',str(s['port'])]
    folder=RUN/'prelaunch/retrievals'/args.tag;folder.mkdir(parents=True,exist_ok=True)
    prepare='R='+repr(REMOTE)+'\nTAG='+repr(args.tag)+'\n'+'''
import pathlib,json,shutil,os
r=pathlib.Path(R);c=pathlib.Path('/workspace/run055-control')
states=[json.loads(p.read_text()) for p in (r/'artifacts/pipeline').glob('*/status.json')]
assert states and all(s['status'] in ('completed','failed') for s in states),'Another pipeline is still writing'
for script in c.glob('h200-components-*.py'):
 assert (c/(script.stem.replace('h200-components-','h200-components-exit-')+'.json')).exists(),'Component check still running'
if (c/'h200-full.py').exists():assert (c/'h200-full-exit.json').exists(),'Full-model preflight still running'
for script in c.glob('h200-diagnosis-*.py'):
 assert (c/(script.stem.replace('h200-diagnosis-','h200-diagnosis-exit-')+'.json')).exists(),'Kernel diagnosis still running'
for p in (r/'prelaunch').glob('calibration-*'):
 if not p.is_dir() or p.name.startswith('calibration-'+TAG):continue
 assert json.loads((p/'result.json').read_text())['status'] in ('passed','failed')
 target=r/'artifacts/preflight-extra'/p.name
 if not target.exists():
  target.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(p,target,copy_function=os.link)
'''
    subprocess.run(ssh+['python3 -'],input=prepare,text=True,check=True)
    # Sealing is refused remotely until the pipeline is terminal. Never reseal a finished archive.
    command=f'test -f {REMOTE}/prelaunch/retrieval-{args.tag}.tar || /workspace/run055-venv/bin/python {REMOTE}/10_seal.py --tag {args.tag}'
    subprocess.run(ssh+[command],check=True)
    receipt_path=folder/'receipt.json'
    subprocess.run(scp+[f'{host}:{REMOTE}/prelaunch/retrieval-{args.tag}.json',str(receipt_path)],check=True)
    receipt=json.loads(receipt_path.read_text());archive=folder/'evidence.tar'
    transfer=receipt['archive'];reused=set()
    if args.reuse_completed_seed:
        assert node['id']=='rjtpxhwiek5tpe' and not args.retired
        prior=json.loads((RUN/'prelaunch/retrieved-completed-seed-001.json').read_text())
        assert prior['status']=='verified' and prior['pod']==node['id']
        prior_rows=json.loads((RUN/'prelaunch/retrievals/completed-seed-001/evidence.json').read_text())['files']
        current={row['path']:row for row in receipt['files']}
        for row in prior_rows:
            assert current[row['path']]==row
            path=RUN/row['path'];assert path.stat().st_size==row['bytes'] and digest(path)==row['sha256']
            reused.add(row['path'])
        wanted=[row['path'] for row in receipt['files'] if row['path'] not in reused]
        code='R='+repr(REMOTE)+'\nTAG='+repr(args.tag)+'\nWANTED='+repr(wanted)+'\n'+'''
import pathlib,json,hashlib,tarfile
r=pathlib.Path(R);receipt=json.loads((r/'prelaunch'/('retrieval-'+TAG+'.json')).read_text())
assert set(WANTED)<={row['path'] for row in receipt['files']}
archive=r/'prelaunch'/('retrieval-'+TAG+'-delta.tar')
with tarfile.open(archive,'w') as a:
 for name in WANTED:
  path=(r/name).resolve();assert path.is_relative_to(r.resolve())
  a.add(path,arcname=name,recursive=False)
with archive.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
print(json.dumps(dict(path=archive.name,bytes=archive.stat().st_size,sha256=digest,files=len(WANTED))))
'''
        transfer=json.loads(subprocess.run(ssh+['python3 -'],input=code,text=True,capture_output=True,check=True).stdout)
        (folder/'transfer-receipt.json').write_text(json.dumps(transfer,indent=2)+'\n')
    if not archive.exists() or archive.stat().st_size!=transfer['bytes'] or digest(archive)!=transfer['sha256']:
        subprocess.run(scp+[f'{host}:{REMOTE}/prelaunch/'+transfer['path'],str(archive)],check=True)
    assert archive.stat().st_size==transfer['bytes'] and digest(archive)==transfer['sha256']
    extracted=folder/'files'
    # Lifecycle evidence has deeply nested checkpoint paths on Windows.
    if os.name=='nt':extracted=Path('\\\\?\\'+str(extracted.resolve()))
    extracted.mkdir(exist_ok=True)
    with tarfile.open(archive) as handle:handle.extractall(extracted,filter='data')
    for record in receipt['files']:
        base=RUN if record['path'] in reused else extracted
        path=(base/record['path']).resolve()
        assert path.is_relative_to(base.resolve())
        assert path.stat().st_size==record['bytes'] and digest(path)==record['sha256'],record['path']
    merged=0
    for record in receipt['files']:
        relative=record['path']
        # Preserve retired worker progress in its isolated archive, avoiding the successful retry's path.
        if args.retired and relative.startswith('artifacts/workers/'):continue
        target_relative=relative
        if args.retired and relative.startswith('artifacts/attempts/'):
            target_relative=relative.replace('artifacts/attempts/','artifacts/retired-attempts/',1)
        source=extracted/relative;target=(RUN/target_relative).resolve()
        assert target.is_relative_to(RUN.resolve())
        if os.name=='nt':target=Path('\\\\?\\'+str(target))
        if target.exists():
            assert target.stat().st_size==record['bytes'] and digest(target)==record['sha256'],relative
        else:
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        merged+=1
    # Explicit allowlist: no SSH keys, credentials, environment variables, or caches.
    code="""import pathlib,tarfile,hashlib,json
r=pathlib.Path('/workspace/run055-control')
files=[p for p in r.iterdir() if p.is_file() and (p.suffix in ('.log','.json') or p.name=='pip-freeze.txt')]
with tarfile.open(r/'control-evidence.tar','w') as a:
 for p in files:a.add(p,arcname=p.name,recursive=False)
with (r/'control-evidence.tar').open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
print(json.dumps(dict(bytes=(r/'control-evidence.tar').stat().st_size,sha256=digest)))
"""
    control=json.loads(subprocess.run(ssh+['python3 -'],input=code,text=True,capture_output=True,check=True).stdout)
    subprocess.run(scp+[f'{host}:/workspace/run055-control/control-evidence.tar',str(folder/'control-evidence.tar')],check=True)
    assert (folder/'control-evidence.tar').stat().st_size==control['bytes'] and digest(folder/'control-evidence.tar')==control['sha256']
    control_dest=RUN/'prelaunch/cloud-controls'/args.tag;control_dest.mkdir(parents=True,exist_ok=True)
    with tarfile.open(folder/'control-evidence.tar') as handle:handle.extractall(control_dest,filter='data')
    result=dict(status='verified',pod=node['id'],tag=args.tag,utc=datetime.now(timezone.utc).isoformat(),
        file_count=len(receipt['files']),merged_files=merged,bytes=receipt['total_bytes'],
        archive_sha256=transfer['sha256'],source_archive_sha256=receipt['archive']['sha256'],
        reused_preverified_files=len(reused),transferred_bytes=transfer['bytes'],
        control_archive=control,retired=args.retired)
    (RUN/'prelaunch'/('retrieved-'+args.tag+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
