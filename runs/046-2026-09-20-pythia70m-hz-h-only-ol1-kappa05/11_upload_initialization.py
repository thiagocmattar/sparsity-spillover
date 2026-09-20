"""Stream the approved random tensors; release preflight only after remote hashes."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import socket
import time

RUN=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('run046_remote',RUN/'10_remote.py')
remote=importlib.util.module_from_spec(spec);spec.loader.exec_module(remote)


def main():
    client,_=remote.connect('training')
    transport=client.get_transport()
    transport.sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
    transport.sock.setsockopt(socket.SOL_SOCKET,socket.SO_SNDBUF,4*1024**2)
    records=[];started=time.monotonic()
    try:
        for name in ['pythia70m-seed1234.safetensors','pythia70m-seed1234-rng.pt']:
            local=RUN/'prelaunch/initialization'/name
            dest=remote.REMOTE+'/runs/'+RUN.name+'/prelaunch/initialization/'+name
            digest=hashlib.sha256(local.read_bytes()).hexdigest()
            channel=transport.open_session(window_size=64*1024**2,max_packet_size=1024**2)
            channel.settimeout(120);channel.exec_command('cat > '+shlex.quote(dest+'.part'))
            done=0;last=0
            with local.open('rb') as handle:
                while chunk:=handle.read(1024**2):
                    channel.sendall(chunk);done+=len(chunk)
                    if time.monotonic()-last>30:
                        print(json.dumps({'file':name,'bytes':done,'total':local.stat().st_size}),flush=True);last=time.monotonic()
            channel.shutdown_write();assert channel.recv_exit_status()==0;channel.close()
            actual=remote.execute(client,'sha256sum '+shlex.quote(dest+'.part')).split()[0]
            assert actual==digest
            remote.execute(client,'mv '+shlex.quote(dest+'.part')+' '+shlex.quote(dest))
            records.append({'name':name,'bytes':done,'sha256':digest})
        remote.execute(client,'touch /workspace/run046-control/initialization-ready')
        result={'files':records,'seconds':time.monotonic()-started,'remote_hashes_verified':True}
        (RUN/'prelaunch/initialization-upload-001.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
    finally:client.close()


if __name__=='__main__':main()
