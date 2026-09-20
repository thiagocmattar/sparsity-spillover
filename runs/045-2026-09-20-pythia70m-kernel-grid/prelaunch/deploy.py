"""Transfer the frozen payload and overlap pinned runtime setup; no model changes."""
import json
from pathlib import Path
import shlex
import time
import remote
import paramiko
import argparse
import subprocess

RUN = Path(__file__).resolve().parent.parent
DEST = '/workspace/run045'
CONTROL = '/workspace/run045-control'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--native-resume', action='store_true')
    args = parser.parse_args()
    client = remote.connect(refresh=True)
    lease = json.loads((RUN/'prelaunch/lease-002.json').read_text())
    started = time.monotonic()
    try:
        remote.execute(client, f'mkdir -p {DEST}/provenance {CONTROL}')
        with paramiko.SFTPClient.from_transport(client.get_transport(), window_size=128*1024*1024, max_packet_size=1024*1024) as sftp:
            for source, target in ([] if args.native_resume else [
                (RUN/'prelaunch/environment_setup.sh', DEST+'/environment_setup.sh'),
                (RUN/'provenance/pip-freeze.txt', DEST+'/provenance/pip-freeze.txt'),
                (RUN/'prelaunch/workload_guard.py', CONTROL+'/workload_guard.py')]):
                sftp.put(str(source), target)
            settings = CONTROL+'/guard-settings.json'
            with sftp.open(settings, 'w') as stream:
                stream.write(json.dumps(dict(pod_id=lease['pod']['id'], name=lease['pod']['name'],
                    deadline_epoch=lease['deadline_epoch'], control=CONTROL)))
            if not args.native_resume:
                remote.execute(client, f'nohup python3 -u {CONTROL}/workload_guard.py {settings} > {CONTROL}/guard.log 2>&1 < /dev/null &')
            command = f'bash {DEST}/environment_setup.sh; code=$?; printf "%s\\n" "$code" > {CONTROL}/environment.exit'
            if not args.native_resume:
                remote.execute(client, f'nohup setsid bash -c {shlex.quote(command)} > {CONTROL}/environment.log 2>&1 < /dev/null &')
            receipt = json.loads((RUN/'bundles/receipt-001.json').read_text())
            if args.native_resume:
                info = json.loads((RUN/'prelaunch/ssh-002.json').read_text())
                print('Starting native SCP of the unchanged frozen bundle', flush=True)
                subprocess.run(['scp', '-O', '-q', '-o', 'BatchMode=yes', '-o',
                    f'UserKnownHostsFile={RUN / "prelaunch/known_hosts"}',
                    '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=20',
                    '-i', info['ssh_key']['path'], '-P', str(info['port']),
                    str(RUN/receipt['path']),
                    f'root@{info["ip"]}:/workspace/run045-input-001.tar.gz'], check=True,
                    timeout=max(1, lease['deadline_epoch']-time.time()-2400))
                print('Native transfer complete; launching hash-checked pipeline', flush=True)
            else:
                sftp.put(str(RUN/receipt['path']), '/workspace/run045-input-001.tar.gz', callback=remote.progress())
            sftp.put(str(RUN/'bundles/inventory-001.json'), CONTROL+'/input-inventory.json')
            script = f'''#!/usr/bin/env bash
set -euo pipefail
cd {DEST}
printf '%s  %s\\n' {receipt['sha256']} /workspace/run045-input-001.tar.gz | sha256sum -c -
tar -xzf /workspace/run045-input-001.tar.gz -C {DEST}
while test ! -f {CONTROL}/environment.exit; do sleep 10; done
test "$(cat {CONTROL}/environment.exit)" = 0
export RUN045_DEADLINE_EPOCH={lease['deadline_epoch']}
runtime/venv/bin/python 01_prepare.py verify
touch runtime/setup-complete
bash 05_execute.sh 03_execute.py --phase preflight --tag 001 --deadline-epoch {lease['deadline_epoch']}
bash 05_execute.sh 03_execute.py --phase final --tag 001 --deadline-epoch {lease['deadline_epoch']}
runtime/venv/bin/python 06_reduce.py --tag 001
runtime/venv/bin/python 08_collect.py --tag 001
'''
            (RUN/'prelaunch/pipeline-001.sh').write_text(script, newline='\n')
            with sftp.open(CONTROL+'/pipeline-001.sh', 'w') as stream:
                stream.write(script)
            command = f'bash {CONTROL}/pipeline-001.sh; code=$?; printf "%s\\n" "$code" > {CONTROL}/pipeline.exit'
            remote.execute(client, f'nohup setsid bash -c {shlex.quote(command)} > {CONTROL}/pipeline.log 2>&1 < /dev/null & echo $! > {CONTROL}/pipeline.pid')
        record = {'status': 'uploaded_pipeline_started', 'pod_id': lease['pod']['id'],
                  'seconds': time.monotonic()-started, 'archive': receipt,
                  'remote_hash_verification': 'first pipeline stage', 'pipeline': 'detached',
                  'deadline_epoch': lease['deadline_epoch']}
        (RUN/'prelaunch/upload-002.json').write_text(json.dumps(record, indent=2)+'\n')
        print(json.dumps(record), flush=True)
    finally:
        client.close()


if __name__ == '__main__':
    main()
