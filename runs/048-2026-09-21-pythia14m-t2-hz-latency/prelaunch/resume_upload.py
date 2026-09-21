"""Infrastructure retry002: retain slow SCP transfer; stream the identical bundle."""
import json
import hashlib
from pathlib import Path
import shlex
import socket
import time
import remote

RUN = Path(__file__).resolve().parent.parent
DEST = '/workspace/run048'
CONTROL = '/workspace/run048-control'
lease = json.loads((RUN/'prelaunch/lease-001.json').read_text())
receipt = json.loads((RUN/'bundles/receipt-001.json').read_text())
assert lease['pod']['name'] == 'run048-14m-t2-hz-001'
client = remote.connect()
started = time.monotonic()
try:
    remote.execute(client, f'test ! -e {CONTROL}/pipeline.pid; test -f {CONTROL}/guard.log')
    remote_path = '/workspace/run048-input-002.tar.gz'
    offset = int(remote.execute(client, f'if test -f {remote_path}; then stat -c%s {remote_path}; else echo 0; fi').strip())
    assert 0 <= offset <= receipt['bytes']
    if offset:
        digest = hashlib.sha256()
        remaining = offset
        with (RUN/receipt['path']).open('rb') as stream:
            while remaining:
                chunk = stream.read(min(1024**2, remaining))
                digest.update(chunk)
                remaining -= len(chunk)
        assert remote.execute(client, f'sha256sum {remote_path}').split()[0] == digest.hexdigest()
    print({'verified_resume_offset': offset, 'remaining_bytes':receipt['bytes']-offset}, flush=True)
    transport = client.get_transport()
    transport.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    transport.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4*1024**2)
    channel = transport.open_session(window_size=64*1024**2, max_packet_size=1024**2)
    channel.settimeout(120)
    channel.exec_command(f'cat >> {remote_path}')
    done = offset
    last = started
    with (RUN/receipt['path']).open('rb') as stream:
        stream.seek(offset)
        while chunk := stream.read(1024**2):
            channel.sendall(chunk)
            done += len(chunk)
            now = time.monotonic()
            if now-last >= 30 or done == receipt['bytes']:
                print({'uploaded_bytes':done, 'total':receipt['bytes'],
                       'MB_per_second':(done-offset)/1e6/(now-started)}, flush=True)
                last = now
    channel.shutdown_write()
    assert channel.recv_exit_status() == 0
    channel.close()
    assert remote.execute(client, 'sha256sum /workspace/run048-input-002.tar.gz').split()[0] == receipt['sha256']
    script = f'''#!/usr/bin/env bash
set -euo pipefail
cd {DEST}
tar -xzf /workspace/run048-input-002.tar.gz -C {DEST}
while test ! -f {CONTROL}/environment.exit; do sleep 10; done
test "$(cat {CONTROL}/environment.exit)" = 0
export RUN048_DEADLINE_EPOCH={lease['deadline_epoch']}
runtime/venv/bin/python 01_prepare.py verify
runtime/venv/bin/python 09_archive_cache.py
touch runtime/setup-complete
bash 05_execute.sh 06_cuda_checks.py
bash 05_execute.sh 03_execute.py --phase smoke --tag 001 --deadline-epoch {lease['deadline_epoch']}
bash 05_execute.sh 03_execute.py --phase final --tag 001 --deadline-epoch {lease['deadline_epoch']}
runtime/venv/bin/python 07_reduce.py --tag 001
runtime/venv/bin/python 08_collect.py --tag 001
'''
    (RUN/'prelaunch/pipeline-002.sh').write_text(script, encoding='utf-8', newline='\n')
    with client.open_sftp() as sftp:
        with sftp.open(CONTROL+'/pipeline-002.sh', 'w') as stream:
            stream.write(script)
    command = f'bash {CONTROL}/pipeline-002.sh; code=$?; printf "%s\\n" "$code" > {CONTROL}/pipeline.exit'
    remote.execute(client, f'nohup setsid bash -c {shlex.quote(command)} > {CONTROL}/pipeline.log 2>&1 < /dev/null & echo $! > {CONTROL}/pipeline.pid')
    result = {'status':'verified_upload_pipeline_started', 'pod_id':lease['pod']['id'],
              'bundle':receipt, 'resume_offset':offset, 'seconds':time.monotonic()-started,
              'retry':'SSH streaming replaces slow SCP; same input bytes and scientific protocol'}
    (RUN/'prelaunch/upload-002.json').write_text(json.dumps(result,indent=2)+'\n', encoding='utf-8', newline='\n')
    print(result, flush=True)
finally:
    client.close()
