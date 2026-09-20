"""Owned-Pod deadline, concurrent environment install and verified input transfer."""
import json
from pathlib import Path
import socket
import time
import remote

RUN = Path(__file__).resolve().parent.parent


def main():
    lease = json.loads((RUN/'prelaunch/lease-001.json').read_text())
    client = remote.connect(refresh=True)
    remote.execute(client, 'mkdir -p /workspace/run042/runtime /workspace/run042/provenance')
    sftp = client.open_sftp()
    for name in ('setup-runtime-only.sh','provenance/pip-freeze.txt'):
        sftp.put(str(RUN/name), '/workspace/run042/'+name)
    remote.execute(client, 'cd /workspace/run042; nohup bash -c '
                   "'bash setup-runtime-only.sh >runtime/setup-001.log 2>&1; echo $? >runtime/setup-001.exit' "
                   '>runtime/setup-launch.log 2>&1 </dev/null &')
    receipt = json.loads((RUN/'bundles/input-001-receipt.json').read_text())
    client.get_transport().sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
    client.get_transport().sock.setsockopt(socket.SOL_SOCKET,socket.SO_SNDBUF,4*1024**2)
    channel = client.get_transport().open_session(window_size=64*1024**2,max_packet_size=1024**2)
    channel.settimeout(120)
    channel.exec_command('cat > /workspace/run042-input-001.tar.gz')
    started = last = time.monotonic()
    done = 0
    with (RUN/receipt['path']).open('rb') as source:
        while chunk := source.read(1024**2):
            channel.sendall(chunk); done += len(chunk)
            now = time.monotonic()
            if now-last >= 30:
                print(json.dumps({'bytes':done,'total':receipt['bytes'],'MBps':done/1e6/(now-started)}),flush=True)
                last = now
    channel.shutdown_write()
    assert channel.recv_exit_status() == 0
    print(remote.execute(client,"cd /workspace; echo '"+receipt['sha256']+"  run042-input-001.tar.gz' | sha256sum -c - && "
                         'tar -xzf run042-input-001.tar.gz -C run042',timeout=180),flush=True)
    print(remote.execute(client,'cd /workspace/run042; python3 01_prepare.py verify',timeout=90),flush=True)
    (RUN/'prelaunch/upload-001.json').write_text(json.dumps({'archive':receipt,'seconds':time.monotonic()-started,
                                                           'verified':True},indent=2)+'\n')
    client.close()


if __name__ == '__main__': main()
