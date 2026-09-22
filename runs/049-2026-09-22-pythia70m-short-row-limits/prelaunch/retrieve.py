"""Retrieve terminal evidence and verify every member before releasing the Pod."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import remote

RUN=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(RUN))
from io_utils import fs


def sha(path):
    h=hashlib.sha256()
    with fs(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',default='001');args=parser.parse_args()
    assert args.tag.isalnum()
    directory=RUN/'retrieval'/args.tag
    directory.mkdir(parents=True,exist_ok=True)
    client=remote.connect()
    try:
        # Packaging itself rejects a live controller; check again before retrieving.
        remote.execute(client,'test ! -f /workspace/run049/artifacts/controller.lock')
        with client.open_sftp() as sftp:
            for name in (f'receipt-{args.tag}.json',f'inventory-{args.tag}.json'):
                sftp.get('/workspace/run049/transfer/'+name,str(directory/name))
            for name in ('environment.log','guard.log','pipeline.log','pipeline.exit','tail.log','tail.exit',
                         'local-runtime-cache.json','local-runtime-cache.log','local-runtime-cache.exit',
                         'local-archive-cache.json','local-archive-cache.log','local-archive-cache.exit'):
                try:sftp.get('/workspace/run049-control/'+name,str(directory/name))
                except FileNotFoundError:pass
    finally:client.close()
    receipt=json.loads((directory/f'receipt-{args.tag}.json').read_text())
    info=json.loads((RUN/'prelaunch/ssh-001.json').read_text())
    archive=directory/Path(receipt['path']).name
    subprocess.run(['scp','-O','-q','-o','BatchMode=yes','-o',
        f'UserKnownHostsFile={RUN / "prelaunch/known_hosts"}', '-o','StrictHostKeyChecking=yes',
        '-i',info['ssh_key']['path'],'-P',str(info['port']),
        f'root@{info["ip"]}:/workspace/run049/{receipt["path"]}',str(archive)],check=True,timeout=600)
    assert archive.stat().st_size==receipt['bytes'] and sha(archive)==receipt['sha256']
    extracted=directory/'extracted'
    with tarfile.open(archive,'r:gz') as tar:
        for item in tar.getmembers():
            relative=PurePosixPath(item.name)
            assert item.isfile() and not relative.is_absolute() and '..' not in relative.parts
            target=fs(extracted/Path(*relative.parts))
            target.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(item) as source,target.open('wb') as output:shutil.copyfileobj(source,output)
    inventory=json.loads((directory/f'inventory-{args.tag}.json').read_text())
    for item in inventory['files']:
        path=fs(extracted/item['path'])
        assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],item['path']
    # Existing scientific inputs are immutable; compare them rather than overwrite.
    for item in inventory['files']:
        source=fs(extracted/item['path']);target=fs(RUN/item['path'])
        if target.exists():assert sha(target)==item['sha256'],f'Local collision: {item["path"]}'
        else:
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    transfer=RUN/'transfer';transfer.mkdir(exist_ok=True)
    for name in (f'receipt-{args.tag}.json',f'inventory-{args.tag}.json'):
        shutil.copy2(directory/name,transfer/name)
    if (directory/'local-runtime-cache.json').exists():
        shutil.copy2(directory/'local-runtime-cache.json',RUN/'provenance/local-runtime-cache.json')
    if (directory/'local-archive-cache.json').exists():
        shutil.copy2(directory/'local-archive-cache.json',RUN/'provenance/local-archive-cache.json')
    result=dict(status='verified',files=len(inventory['files']),archive=receipt,
                pod_id=info['id'],tag=args.tag,existing_sources_preserved=True)
    (RUN/'results').mkdir(exist_ok=True)
    (RUN/'results/local-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
