"""Retrieve and verify one completed Run 032 worker before its Pod is deleted."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('run032_remote',HERE/'07_remote.py')
remote=importlib.util.module_from_spec(spec)
spec.loader.exec_module(remote)


def retrieve(pod, archive_path=None):
    dest=HERE/'retrieval'/pod['id']
    dest.mkdir(parents=True,exist_ok=True)
    archive=Path(archive_path) if archive_path else dest/'results.tar'
    evidence=HERE/'prelaunch/cloud'/pod['id']
    evidence.mkdir(parents=True,exist_ok=True)
    with remote.connect(pod['id']) as client:
        exitcode=remote.command(client,'cat /workspace/run032-training.exit').strip()
        if exitcode!='0':raise RuntimeError('Training, verification or packaging failed: '+exitcode)
        expected=remote.command(client,'cat /workspace/run032-results.sha256').split()[0]
        with client.open_sftp() as sftp:
            if archive_path is None:
                sftp.get('/workspace/run032-results.tar',str(archive),prefetch=True,max_concurrent_prefetch_requests=64)
            sources=[f'{remote.REMOTE}/{remote.RUN}/prelaunch/remote-preflight.json']
            sources += ['/workspace/'+n for n in sftp.listdir('/workspace') if n.startswith('run032-') and (n.endswith('.log') or n.endswith('.exit') or n.endswith('.sha256'))]
            for source in sources:
                sftp.get(source,str(evidence/Path(source).name))
    actual=remote.sha(archive)
    if actual!=expected:raise RuntimeError('Archive transfer hash mismatch')
    with tarfile.open(archive) as handle:
        for member in handle.getmembers():
            target=(HERE/member.name).resolve()
            if not target.is_relative_to((HERE/'artifacts').resolve()) or member.issym() or member.islnk():
                raise ValueError('Unsafe artifact archive member: '+member.name)
            if member.isfile() and target.exists():
                raise FileExistsError('Never overwrite an existing scientific artifact: '+str(target))
        handle.extractall(HERE,filter='data')
    subprocess.run([sys.executable,str(HERE/'03_verify.py'),'--condition',pod['condition']],check=True)
    receipt={'pod':pod['id'],'condition':pod['condition'],'retrieved_utc':datetime.now(timezone.utc).isoformat(),'archive_bytes':archive.stat().st_size,'remote_sha256':expected,'local_sha256':actual,'standalone_verification':'passed'}
    (evidence/'retrieval-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--pod',required=True)
    args=parser.parse_args()
    pods=json.loads((HERE/'prelaunch/pods.json').read_text(encoding='utf-8-sig'))
    retrieve(next(p for p in pods if p['id']==args.pod))
