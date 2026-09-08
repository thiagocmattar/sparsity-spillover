"""Retrieve completed Run 031 attempts and verify every transferred byte."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pod-id', required=True)
    args = parser.parse_args()
    control = HERE/'launch-control/worker-001'
    connection = json.loads((control/'connection.json').read_text(encoding='utf-8-sig'))
    assert connection['id'] == args.pod_id
    options = ['-i',connection['ssh_key']['path'],'-o','StrictHostKeyChecking=yes',
               '-o','UserKnownHostsFile='+str(control/'known_hosts')]
    ssh = ['ssh',*options,'-p',str(connection['port']),f'root@{connection["ip"]}']
    script = r'''
import hashlib, io, json, pathlib, tarfile
root=pathlib.Path('/workspace/run031')
run=root/'runs/031-2026-09-08-signed-activation-density'
for name in ('001-runpod-smoke','002-runpod-full'):
    manifest=json.loads((run/'artifacts/attempts'/name/'manifest.json').read_text())
    assert manifest['status']=='completed',name
files={p.relative_to(run/'artifacts').as_posix():p.read_bytes()
       for p in sorted((run/'artifacts').rglob('*')) if p.is_file()}
files['setup.log']=(root/'setup.log').read_bytes()
inventory={name:{'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest()}
           for name,blob in files.items()}
files['transfer-inventory.json']=(json.dumps(inventory,indent=2)+'\n').encode()
archive=root/'signed-density-export.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for name,blob in files.items():
        info=tarfile.TarInfo(name);info.size=len(blob)
        tar.addfile(info,io.BytesIO(blob))
print(json.dumps({'bytes':archive.stat().st_size,
                  'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files':len(inventory)}))
'''
    expected = json.loads(subprocess.check_output(ssh+['python -'],input=script.encode()))
    target = HERE/'artifacts/retrieved'
    target.mkdir(parents=True,exist_ok=False)
    archive = target/'signed-density-export.tar.gz'
    subprocess.run(['scp',*options,'-P',str(connection['port']),
                    f'root@{connection["ip"]}:/workspace/run031/signed-density-export.tar.gz',str(archive)],check=True)
    assert archive.stat().st_size == expected['bytes']
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == expected['sha256']
    with tarfile.open(archive) as tar:
        inventory = json.load(tar.extractfile('transfer-inventory.json'))
        for name,entry in inventory.items():
            destination = (target/name).resolve()
            assert destination.is_relative_to(target.resolve())
            blob = tar.extractfile(name).read()
            assert len(blob)==entry['bytes'] and hashlib.sha256(blob).hexdigest()==entry['sha256']
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(blob)
    receipt = {**expected,'pod_id':args.pod_id,'retrieved_utc':datetime.now(timezone.utc).isoformat(),
               'files_verified':len(inventory),'inventory':inventory}
    (HERE/'transfer-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k!='inventory'}))


if __name__ == '__main__':
    main()
