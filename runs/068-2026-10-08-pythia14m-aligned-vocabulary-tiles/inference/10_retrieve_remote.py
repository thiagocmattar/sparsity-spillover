"""Download the sealed benchmark evidence through the owned Pod's SSH stream."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import socket
import time

RUN = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('head_transport', RUN / '09_remote.py')
transport_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport_module)


def main():
    client, lease = transport_module.remote.connect('benchmark')
    destination = RUN / 'transfer'
    destination.mkdir(exist_ok=True)
    remote_root = '/workspace/run068-head/runs/' + RUN.parent.name + '/inference'
    started = time.monotonic()
    try:
        if transport_module.remote.execute(client, 'cat /workspace/run068-control/pipeline.exit').strip() != '0':
            raise ValueError('Successful detached pipeline required')
        with client.open_sftp() as sftp:
            for name, remote_path in (
                ('seal-001.json', remote_root + '/transfer/seal-001.json'),
                ('pipeline-benchmark-001.log', '/workspace/run068-control/pipeline.log'),
                ('pipeline-benchmark-001.exit', '/workspace/run068-control/pipeline.exit'),
            ):
                target = destination / name if name == 'seal-001.json' else RUN / 'prelaunch' / name
                with sftp.open(remote_path, 'rb') as handle:
                    payload = handle.read()
                if target.exists() and target.read_bytes() != payload:
                    raise ValueError('Existing local evidence differs: ' + name)
                target.write_bytes(payload)
        seal = json.loads((destination / 'seal-001.json').read_text())
        expected = seal['archive']
        assert expected['path'] == 'transfer/results-001.tar.gz'
        target = destination / 'results-001.tar.gz'
        if not target.exists():
            partial = target.with_suffix('.gz.part')
            offset = partial.stat().st_size if partial.exists() else 0
            assert 0 <= offset <= expected['bytes']
            transport = client.get_transport()
            transport.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            transport.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024**2)
            channel = transport.open_session(window_size=64 * 1024**2, max_packet_size=1024**2)
            channel.settimeout(120)
            channel.exec_command(f'tail -c +{offset + 1} ' + shlex.quote(remote_root + '/' + expected['path']))
            with partial.open('ab') as handle:
                while chunk := channel.recv(1024**2):
                    handle.write(chunk)
            assert channel.recv_exit_status() == 0, 'SSH stream failed; partial retained'
            channel.close()
            assert partial.stat().st_size == expected['bytes']
            with partial.open('rb') as handle:
                assert hashlib.file_digest(handle, 'sha256').hexdigest() == expected['sha256']
            partial.replace(target)
        with target.open('rb') as handle:
            assert target.stat().st_size == expected['bytes']
            assert hashlib.file_digest(handle, 'sha256').hexdigest() == expected['sha256']
        print(json.dumps({'status': 'archive-downloaded-and-hash-verified', 'pod_id': lease['pod']['id'],
                          'archive': expected, 'seconds': time.monotonic() - started,
                          'member_verification_required': True}), flush=True)
    finally:
        client.close()


if __name__ == '__main__':
    main()
