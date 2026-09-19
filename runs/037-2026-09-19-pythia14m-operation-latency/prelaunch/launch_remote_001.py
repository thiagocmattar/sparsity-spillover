"""Infrastructure-only upload, credential-isolated stop guard and detached setup."""
import io
import json
import os
from pathlib import Path
import time
import tomllib
import remote

RUN = Path(__file__).resolve().parent.parent


def main():
    lease = json.loads((RUN / 'prelaunch/lease-001.json').read_text())
    receipt = json.loads((RUN / 'bundles/input-001.receipt.json').read_text())
    client = remote.connect()
    sftp = client.open_sftp()
    started = time.monotonic()
    sftp.put(str(RUN / receipt['path']), '/workspace/run037-input-001.tar', callback=remote.progress())
    transfer = {'bytes': receipt['bytes'], 'seconds': time.monotonic() - started}
    print(remote.execute(client,
        "cd /workspace && echo '" + receipt['sha256'] + "  run037-input-001.tar' | sha256sum -c - "
        "&& mkdir -p run037 && tar -xf run037-input-001.tar -C run037 && mkdir -p run037/runtime"))
    key = os.environ.get('RUNPOD_API_KEY')
    if not key:
        key = tomllib.loads((Path.home() / '.runpod/config.toml').read_text())['apikey']
    settings = {'api_key': key, 'pod_id': lease['pod']['id'], 'name': lease['pod']['name'],
                'deadline_epoch': lease['deadline_epoch']}
    secret = '/workspace/run037/runtime/guard-settings-key.json'
    with sftp.open(secret, 'w') as handle: handle.write(json.dumps(settings))
    sftp.chmod(secret, 0o600)
    del key, settings
    print(remote.execute(client,
        'cd /workspace/run037 && setsid python3 -u 10_deadline_guard.py runtime/guard-settings-key.json '
        '> runtime/deadline-guard.log 2>&1 < /dev/null &'))
    command = ('cd /workspace/run037 && setsid bash -c '
               "'bash 04_setup.sh > runtime/setup-001.log 2>&1; code=$?; printf \"%s\\n\" \"$code\" "
               "> runtime/setup-001.exit' < /dev/null > runtime/setup-launch.log 2>&1 &")
    remote.execute(client, command)
    (RUN / 'prelaunch/upload-001.json').write_text(json.dumps(transfer, indent=2) + '\n')
    print(json.dumps({'uploaded_and_hash_verified': True, 'transfer': transfer,
                      'setup': 'detached', 'guard': 'started'}), flush=True)
    sftp.close(); client.close()


if __name__ == '__main__': main()
