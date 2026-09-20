"""Infrastructure-only transfer, scoped stop guard, and detached environment setup."""
import json
import os
from pathlib import Path
import time
import tomllib
import remote

RUN = Path(__file__).resolve().parent.parent


def main():
    lease = json.loads((RUN / 'prelaunch/lease-001.json').read_text())
    receipt = json.loads((RUN / 'bundles/input-003.receipt.json').read_text())
    client = remote.connect()
    sftp = client.open_sftp()
    remote.execute(client, 'mkdir -p /workspace/run040/runtime')
    sftp.put(str(RUN / '12_deadline_guard.py'), '/workspace/run040/12_deadline_guard.py')
    # Already armed before the upload; preserve the live guard and its consumed secret.
    print(remote.execute(client, 'cat /workspace/run040/runtime/deadline-guard.log; '
                        'test ! -e /workspace/run040/runtime/guard-settings-key.json'), flush=True)
    started = time.monotonic()
    sftp.put(str(RUN / receipt['path']), '/workspace/run040-input-003.tar', callback=remote.progress())
    transfer = {'bytes': receipt['bytes'], 'seconds': time.monotonic() - started}
    print(remote.execute(client,
        "cd /workspace && echo '" + receipt['sha256'] + "  run040-input-003.tar' | sha256sum -c - "
        "&& tar -xf run040-input-003.tar -C run040"), flush=True)
    command = ('cd /workspace/run040; nohup bash -c '
               "'bash 04_setup.sh > runtime/setup-001.log 2>&1; code=$?; printf \"%s\\n\" \"$code\" "
               "> runtime/setup-001.exit' < /dev/null > runtime/setup-launch.log 2>&1 & echo setup-launched")
    remote.execute(client, command)
    (RUN / 'prelaunch/upload-001.json').write_text(json.dumps(transfer, indent=2) + '\n')
    print(json.dumps({'uploaded_and_hash_verified': True, 'transfer': transfer,
                      'setup': 'detached', 'guard': 'armed'}), flush=True)
    sftp.close()
    client.close()


if __name__ == '__main__':
    main()
