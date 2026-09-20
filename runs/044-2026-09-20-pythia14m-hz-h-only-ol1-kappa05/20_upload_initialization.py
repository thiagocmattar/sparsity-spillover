"""Resumable SSH upload of the existing step-zero bytes; no scientific changes."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import socket
import time

RUN=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('run044_transport',RUN/'10_remote.py')
remote=importlib.util.module_from_spec(spec);spec.loader.exec_module(remote)

def main():
    source=RUN/'inputs/random-initialization/model.safetensors'
    provenance=json.loads((source.parent/'provenance.json').read_text())
    with source.open('rb') as handle:
        assert hashlib.file_digest(handle,'sha256').hexdigest()==provenance['file_sha256']
    client,_=remote.connect('training')
    destination=f'{remote.REMOTE}/runs/{RUN.name}/inputs/random-initialization/model.safetensors'
    partial=destination+'.part'
    transport=client.get_transport()
    transport.sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
    transport.sock.setsockopt(socket.SOL_SOCKET,socket.SO_SNDBUF,4*1024**2)
    try:
        offset=int(remote.execute(client,'stat -c %s '+shlex.quote(partial)).strip())
        assert 0<=offset<=source.stat().st_size
        channel=transport.open_session(window_size=64*1024**2,max_packet_size=1024**2)
        channel.settimeout(120);channel.exec_command('cat >> '+shlex.quote(partial))
        started=time.monotonic();last=started;done=offset
        with source.open('rb') as handle:
            handle.seek(offset)
            while chunk:=handle.read(1024**2):
                channel.sendall(chunk);done+=len(chunk)
                if time.monotonic()-last>=30:
                    print(json.dumps({'bytes':done,'total':source.stat().st_size}),flush=True)
                    last=time.monotonic()
        channel.shutdown_write();assert channel.recv_exit_status()==0;channel.close()
        actual=remote.execute(client,'sha256sum '+shlex.quote(partial)).split()[0]
        assert actual==provenance['file_sha256']
        remote.execute(client,'mv -- '+shlex.quote(partial)+' '+shlex.quote(destination))
        result={'sha256':actual,'bytes':done,'resume_offset':offset,'seconds':time.monotonic()-started,
                'trained_updates':0,'verified':True,'transport':'SSH TCP_NODELAY and 4MiB send buffer'}
        (RUN/'prelaunch/initialization-upload-002.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
    finally:client.close()

if __name__=='__main__':main()
