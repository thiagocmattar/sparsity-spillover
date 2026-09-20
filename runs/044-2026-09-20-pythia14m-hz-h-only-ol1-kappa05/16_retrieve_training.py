"""Retrieve the sealed training archive; verify every file before Pod teardown."""
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import shlex
import tarfile
import time
import paramiko

RUN=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('run044_transport',RUN/'10_remote.py')
remote=importlib.util.module_from_spec(spec);spec.loader.exec_module(remote)


def digest(path):
    with path.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def main():
    staging=RUN/'retrieval';staging.mkdir(exist_ok=True)
    client,_=remote.connect('training')
    transport=client.get_transport()
    transport.sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
    transport.sock.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,8*1024**2)
    sftp=paramiko.SFTPClient.from_transport(transport,window_size=64*1024**2,max_packet_size=1024**2)
    started=time.monotonic();last=[0.]
    def progress(done,total):
        if time.monotonic()-last[0]>30 or done==total:
            last[0]=time.monotonic()
            print(json.dumps({'bytes':done,'total_bytes':total,
                'MiB_per_second':done/1024**2/max(.01,time.monotonic()-started)}),flush=True)
    remote_root=remote.REMOTE+'/runs/'+RUN.name
    receipt_path=staging/'training-receipt-001.json'
    destination=staging/'training-001.tar.gz'
    try:
        sftp.get(remote_root+'/transfer/training-receipt-001.json',str(receipt_path))
        receipt=json.loads(receipt_path.read_text())
        assert receipt['path']=='transfer/training-001.tar.gz'
        if not destination.exists():
            part=destination.with_suffix('.gz.part')
            offset=part.stat().st_size if part.exists() else 0
            assert 0<=offset<=receipt['bytes']
            channel=transport.open_session(window_size=64*1024**2,max_packet_size=1024**2)
            channel.settimeout(120)
            channel.exec_command(f'tail -c +{offset+1} '+shlex.quote(remote_root+'/'+receipt['path']))
            done=offset
            with part.open('ab') as handle:
                while chunk:=channel.recv(1024**2):
                    handle.write(chunk);done+=len(chunk);progress(done,receipt['bytes'])
            assert channel.recv_exit_status()==0,'SSH stream failed; partial file retained'
            channel.close()
            assert part.stat().st_size==receipt['bytes'] and digest(part)==receipt['sha256']
            part.replace(destination)
        assert destination.stat().st_size==receipt['bytes'] and digest(destination)==receipt['sha256']
    finally:sftp.close();client.close()
    extracted=staging/'training-001';extracted.mkdir(exist_ok=True)
    with tarfile.open(destination,'r:gz') as bundle:
        bundle.extractall(extracted,filter='data')
    inventory=json.loads((extracted/'transfer/training-inventory-001.json').read_text())
    for row in inventory['files']:
        path=extracted/row['path']
        if not path.resolve().is_relative_to(extracted.resolve()):raise ValueError('Unscoped path')
        assert path.stat().st_size==row['bytes'] and digest(path)==row['sha256'],row['path']
    # Install verified files only; never replace an existing, different artifact.
    for row in inventory['files']:
        source=extracted/row['path'];target=RUN/row['path']
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert digest(target)==row['sha256'],str(target)
        else:source.replace(target)
    target=RUN/'transfer';target.mkdir(exist_ok=True)
    for source in [extracted/'transfer/training-inventory-001.json',receipt_path]:
        (target/source.name).write_bytes(source.read_bytes())
    result={'archive_and_members_verified':True,'files':len(inventory['files']),
        'archive':receipt,'seconds':time.monotonic()-started,'pod_retained':True}
    (RUN/'prelaunch/retrieval-training-001.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
