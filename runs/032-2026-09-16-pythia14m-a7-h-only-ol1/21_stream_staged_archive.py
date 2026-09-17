"""Stream an already staged archive over an SSH command, avoiding stalled SFTP."""
import argparse
import importlib.util
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import time

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('remote032',HERE/'07_remote.py')
remote=importlib.util.module_from_spec(spec)
spec.loader.exec_module(remote)

def recover(pod):
    dest=HERE/'retrieval'/pod['id']
    dest.mkdir(parents=True,exist_ok=True)
    with remote.connect(pod['id']) as client:
        paths=remote.command(client,'ls /opt/run032-recovery-*.tar').splitlines()
        if len(paths)!=1:raise ValueError('Expected exactly one staged recovery archive')
        source=paths[0]
        expected=remote.command(client,'sha256sum '+shlex.quote(source)).split()[0]
        size=int(remote.command(client,'stat -c %s '+shlex.quote(source)))
        target=dest/('stream-'+Path(source).name)
        offset=target.stat().st_size if target.exists() else 0
        if offset>size:raise ValueError('Oversized local partial')
        started=time.monotonic()
        copied=offset
        last=copied
        _,stdout,stderr=client.exec_command('tail -c +'+str(offset+1)+' '+shlex.quote(source),timeout=60)
        with target.open('ab') as handle:
            while copied<size:
                data=stdout.read(min(1024*1024,size-copied))
                if not data:raise EOFError('Incomplete stream')
                handle.write(data)
                copied+=len(data)
                if copied-last>=50*1024*1024:
                    print(json.dumps({'pod':pod['id'],'bytes':copied,'total':size,'MB_per_s':(copied-offset)/1e6/(time.monotonic()-started)}),flush=True)
                    last=copied
        if stdout.channel.recv_exit_status()!=0:raise RuntimeError(stderr.read().decode())
    assert target.stat().st_size==size and remote.sha(target)==expected
    with tarfile.open(target) as archive:
        for member in archive.getmembers():
            output=(HERE/member.name).resolve()
            if not member.isfile() or not output.is_relative_to((HERE/'artifacts').resolve()):raise ValueError('Unsafe member')
            if output.exists():raise FileExistsError(output)
        archive.extractall(HERE,filter='data')
    subprocess.run([sys.executable,str(HERE/'03_verify.py'),'--condition',pod['condition']],check=True)
    receipt={'pod':pod['id'],'condition':pod['condition'],'standalone_verification':'passed','method':'Staged missing files streamed by SSH command; complete per-file original inventory verified','archive_bytes':size,'archive_sha256':expected,'retrieved_epoch':time.time()}
    (HERE/'prelaunch/cloud'/pod['id']/'retrieval-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--pod',required=True)
    args=parser.parse_args()
    pods=json.loads((HERE/'prelaunch/pods.json').read_text())
    recover(next(p for p in pods if p['id']==args.pod))
