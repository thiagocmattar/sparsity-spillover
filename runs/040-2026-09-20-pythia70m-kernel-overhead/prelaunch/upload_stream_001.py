"""Retry only the transport; compressed bytes expand to the same approved bundle."""
import json
from pathlib import Path
import time
import socket
import remote

RUN = Path(__file__).resolve().parent.parent


def main():
    receipt = json.loads((RUN / 'bundles/input-003-compressed.receipt.json').read_text())
    client = remote.connect()
    client.get_transport().sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    client.get_transport().sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024**2)
    offset = int(remote.execute(client, 'stat -c %s /workspace/run040-input-003.tar.gz'))
    channel = client.get_transport().open_session(window_size=64 * 1024**2, max_packet_size=1024**2)
    channel.settimeout(120)
    channel.exec_command('cat >> /workspace/run040-input-003.tar.gz')
    started = last = time.monotonic()
    done = offset
    with (RUN / receipt['path']).open('rb') as source:
        source.seek(offset)
        while chunk := source.read(1024**2):
            channel.sendall(chunk)
            done += len(chunk)
            now = time.monotonic()
            if now - last > 30:
                print(json.dumps({'bytes': done, 'total': receipt['bytes'], 'MBps': (done-offset) / 1e6 / (now-started)}), flush=True)
                last = now
    channel.shutdown_write()
    assert channel.recv_exit_status() == 0
    transfer = {'bytes': done, 'seconds': time.monotonic()-started, 'method': 'compressed SSH stream'}
    print(remote.execute(client, "cd /workspace; echo '" + receipt['sha256'] + "  run040-input-003.tar.gz' | sha256sum -c - && tar -xzf run040-input-003.tar.gz -C run040", timeout=120), flush=True)
    print(remote.execute(client, 'cd /workspace/run040; nohup bash -c '
          "'bash 04_setup.sh > runtime/setup-001.log 2>&1; code=$?; printf \"%s\\n\" \"$code\" > runtime/setup-001.exit' "
          '< /dev/null > runtime/setup-launch.log 2>&1 & echo setup-launched'), flush=True)
    (RUN / 'prelaunch/upload-stream-001.json').write_text(json.dumps(transfer, indent=2)+'\n')
    print(transfer, flush=True)
    client.close()


if __name__ == '__main__': main()
