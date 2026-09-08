"""Retrieve completed checkpoint outputs and verify the exact transferred bytes."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
from datetime import datetime, timezone

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def main():
    connection=json.loads(subprocess.check_output([
        str(ROOT/'tmp/runpodctl-v2.12.0.exe'),'ssh','info','3bibov10sjbpjw']))
    options=['-i',connection['ssh_key']['path'],'-o','StrictHostKeyChecking=yes',
             '-o','UserKnownHostsFile='+str(HERE/'launch-control/rtx5090-001/known_hosts')]
    ssh=['ssh',*options,'-p',str(connection['port']),f'root@{connection["ip"]}']
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    remote=f'/workspace/run030/export-{stamp}.tar.gz'
    script=r'''
import hashlib,io,json,pathlib,tarfile
root=pathlib.Path('/workspace/run030')
files={}
for worker in ('worker1','worker2'):
 folder=root/'production'/worker
 if not folder.exists(): continue
 for sweep in folder.glob('*/sweep.json'):
  for path in sweep.parent.glob('*.json'):
   name=sweep.parent.name+'/'+path.name
   assert name not in files
   files[name]=path.read_bytes()
 manifest=folder/'manifest.json'
 if manifest.exists() and json.loads(manifest.read_text())['status'] in ('completed','failed'):
  for p in (manifest,folder/'events.jsonl',root/(worker+'-hardware.csv'),root/(worker+'.log')):
   files['workers/'+worker+'/'+p.name]=p.read_bytes()
for name in ('smoke','smoke-worker2'):
 folder=root/name
 manifest=folder/'manifest.json'
 if manifest.exists() and json.loads(manifest.read_text())['status']=='completed':
  for path in folder.rglob('*'):
   if path.is_file():
    files['preflight/'+name+'/'+path.relative_to(folder).as_posix()]=path.read_bytes()
inventory={name:{'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest()} for name,blob in files.items()}
files['transfer_inventory.json']=json.dumps(inventory,indent=2).encode()
with tarfile.open(ARCHIVE,'w:gz') as tar:
 for name,blob in files.items():
  info=tarfile.TarInfo(name);info.size=len(blob)
  tar.addfile(info,io.BytesIO(blob))
print(json.dumps({'archive_sha256':hashlib.sha256(pathlib.Path(ARCHIVE).read_bytes()).hexdigest(),'files':len(inventory),'completed_sweeps':sum(name.endswith('/sweep.json') for name in files)}))
'''.replace('ARCHIVE',repr(remote))
    meta=json.loads(subprocess.check_output(ssh+['python -'],input=script.encode()))
    archive=HERE/'artifacts'/f'export-{stamp}.tar.gz'
    archive.parent.mkdir(exist_ok=True)
    subprocess.run(['scp',*options,'-P',str(connection['port']),f'root@{connection["ip"]}:{remote}',str(archive)],check=True)
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==meta['archive_sha256']
    target=HERE/'artifacts/attempts/001'
    target.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive) as tar:
        inventory=json.load(tar.extractfile('transfer_inventory.json'))
        for name,expected in inventory.items():
            destination=(target/name).resolve()
            assert destination.is_relative_to(target.resolve())
            blob=tar.extractfile(name).read()
            assert len(blob)==expected['bytes'] and hashlib.sha256(blob).hexdigest()==expected['sha256']
            if destination.exists():
                assert destination.read_bytes()==blob, str(destination)
            else:
                destination.parent.mkdir(parents=True,exist_ok=True)
                destination.write_bytes(blob)
    receipt={**meta,'retrieved_utc':datetime.now(timezone.utc).isoformat(),
             'pod_id':connection['id'],'archive':archive.relative_to(ROOT).as_posix(),
             'files_verified':len(inventory),'inventory':inventory}
    (HERE/'artifacts'/f'receipt-{stamp}.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='inventory'}))


if __name__=='__main__':
    main()
