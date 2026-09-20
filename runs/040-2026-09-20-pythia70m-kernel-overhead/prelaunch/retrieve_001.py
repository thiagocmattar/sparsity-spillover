"""Retrieve the sealed output archive and verify every member before teardown."""
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import paramiko
import remote

RUN=Path(__file__).resolve().parent.parent


def main(tag):
    assert tag.isalnum()
    staging=RUN/'retrieval'
    staging.mkdir(exist_ok=True)
    client=remote.connect()
    transport=client.get_transport()
    transport.sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
    transport.sock.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,8*1024**2)
    sftp=paramiko.SFTPClient.from_transport(transport,window_size=32*1024**2,max_packet_size=1024**2)
    started=time.monotonic()
    try:
        receipt_path=staging/f'receipt-{tag}.json'
        assert not receipt_path.exists(), 'Preserve an earlier retrieval attempt'
        sftp.get(f'/workspace/run040/transfer/receipt-{tag}.json',str(receipt_path))
        receipt=json.loads(receipt_path.read_text())
        assert receipt['path']==f'transfer/output-{tag}.tar.gz'
        destination=staging/f'output-{tag}.tar.gz'
        assert not destination.exists()
        part=destination.with_suffix(destination.suffix+'.part')
        sftp.get('/workspace/run040/'+receipt['path'],str(part),callback=remote.progress(),
                 max_concurrent_prefetch_requests=64)
        with part.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
        assert part.stat().st_size==receipt['bytes'] and digest==receipt['sha256']
        part.replace(destination)
    finally:
        sftp.close();client.close()
    subprocess.run([sys.executable,str(RUN/'09_verify_retrieval.py'),'--tag',tag],check=True)
    result={'tag':tag,'archive':receipt,'seconds':time.monotonic()-started,
            'archive_and_all_members_verified':True,'pod_still_retained':True}
    (RUN/'prelaunch'/f'retrieval-{tag}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main(sys.argv[1] if len(sys.argv)>1 else '001')
