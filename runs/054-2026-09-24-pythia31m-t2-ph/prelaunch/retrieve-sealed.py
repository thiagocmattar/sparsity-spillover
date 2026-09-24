"""Retrieve one terminal training Pod; retain and check every byte before teardown."""
import argparse
import hashlib
import json
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
    p.add_argument('--connections',default='parallel-connections-002.json')
    p.add_argument('--node',type=int,required=True)
    p.add_argument('--tag',required=True)
    p.add_argument('--retired',action='store_true')
    args=p.parse_args()
    if not args.tag.replace('-','').isalnum():raise ValueError('Invalid tag')
    node=json.loads((RUN/'prelaunch'/args.connections).read_text())[args.node]
    s=node['ssh'];host=s['username']+'@'+s['host']
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-i',KEY,'-p',str(s['port']),host]
    scp=['scp','-q','-o','BatchMode=yes','-i',KEY,'-P',str(s['port'])]
    folder=RUN/'prelaunch/retrievals'/args.tag;folder.mkdir(parents=True,exist_ok=True)
    # Sealing is refused remotely until the pipeline is terminal. Never reseal a finished archive.
    command=f'test -f {REMOTE}/prelaunch/retrieval-{args.tag}.tar || /workspace/run054-venv/bin/python {REMOTE}/10_seal.py --tag {args.tag}'
    subprocess.run(ssh+[command],check=True)
    receipt_path=folder/'receipt.json'
    subprocess.run(scp+[f'{host}:{REMOTE}/prelaunch/retrieval-{args.tag}.json',str(receipt_path)],check=True)
    receipt=json.loads(receipt_path.read_text());archive=folder/'evidence.tar'
    if not archive.exists() or archive.stat().st_size!=receipt['archive']['bytes'] or digest(archive)!=receipt['archive']['sha256']:
        subprocess.run(scp+[f'{host}:{REMOTE}/prelaunch/'+receipt['archive']['path'],str(archive)],check=True)
    assert archive.stat().st_size==receipt['archive']['bytes'] and digest(archive)==receipt['archive']['sha256']
    extracted=folder/'files';extracted.mkdir(exist_ok=True)
    with tarfile.open(archive) as handle:handle.extractall(extracted,filter='data')
    for record in receipt['files']:
        path=(extracted/record['path']).resolve()
        assert path.is_relative_to(extracted.resolve())
        assert path.stat().st_size==record['bytes'] and digest(path)==record['sha256'],record['path']
    merged=0
    for record in receipt['files']:
        relative=record['path']
        # Preserve retired worker progress in its isolated archive, avoiding the successful retry's path.
        if args.retired and relative.startswith('artifacts/workers/'):continue
        source=extracted/relative;target=(RUN/relative).resolve()
        assert target.is_relative_to(RUN.resolve())
        if target.exists():
            assert target.stat().st_size==record['bytes'] and digest(target)==record['sha256'],relative
        else:
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        merged+=1
    # Explicit allowlist: no SSH keys, credentials, environment variables, or caches.
    code="""import pathlib,tarfile,hashlib,json
r=pathlib.Path('/workspace/run054-control')
files=[p for p in r.iterdir() if p.is_file() and (p.suffix in ('.log','.json') or p.name=='pip-freeze.txt')]
with tarfile.open(r/'control-evidence.tar','w') as a:
 for p in files:a.add(p,arcname=p.name,recursive=False)
with (r/'control-evidence.tar').open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
print(json.dumps(dict(bytes=(r/'control-evidence.tar').stat().st_size,sha256=digest)))
"""
    control=json.loads(subprocess.run(ssh+['python3 -'],input=code,text=True,capture_output=True,check=True).stdout)
    subprocess.run(scp+[f'{host}:/workspace/run054-control/control-evidence.tar',str(folder/'control-evidence.tar')],check=True)
    assert (folder/'control-evidence.tar').stat().st_size==control['bytes'] and digest(folder/'control-evidence.tar')==control['sha256']
    control_dest=RUN/'prelaunch/cloud-controls'/args.tag;control_dest.mkdir(parents=True,exist_ok=True)
    with tarfile.open(folder/'control-evidence.tar') as handle:handle.extractall(control_dest,filter='data')
    result=dict(status='verified',pod=node['id'],tag=args.tag,utc=datetime.now(timezone.utc).isoformat(),
        file_count=len(receipt['files']),merged_files=merged,bytes=receipt['total_bytes'],
        archive_sha256=receipt['archive']['sha256'],control_archive=control,retired=args.retired)
    (RUN/'prelaunch'/('retrieved-'+args.tag+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
