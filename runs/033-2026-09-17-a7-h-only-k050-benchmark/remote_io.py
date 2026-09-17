"""Run-local SSH transport; provider identity comes from the recorded lease."""
import json
from pathlib import Path
import subprocess
import time

import paramiko

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def connect(*, refresh=False):
    lease = json.loads((HERE/'prelaunch/lease-001.json').read_text())
    saved = HERE/'prelaunch/ssh-001.json'
    # The endpoint is stable during this unchanged Pod lifecycle. Refresh after
    # any restart; monitoring need not depend on a healthy provider control API.
    if saved.exists() and not refresh:
        info = json.loads(saved.read_text())
    else:
        info = json.loads(subprocess.check_output([
            str(ROOT/'tmp/runpodctl-v2.12.0.exe'), 'ssh', 'info', lease['pod']['id']], text=True))
    assert info['id'] == lease['pod']['id'] and info['name'] == 'run033-k050-new-001'
    saved.write_text(json.dumps(info, indent=2)+'\n')
    known = HERE/'prelaunch/known_hosts'
    if not known.exists(): known.touch()
    client = paramiko.SSHClient()
    client.load_host_keys(str(known))
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(info['ip'], port=info['port'], username='root',
                   key_filename=info['ssh_key']['path'], timeout=20,
                   auth_timeout=20, banner_timeout=20, look_for_keys=False, allow_agent=False)
    client.get_transport().set_keepalive(20)
    return client


def execute(client, command, timeout=60):
    _, out, err = client.exec_command(command, timeout=timeout)
    stdout = out.read().decode(); stderr = err.read().decode()
    code = out.channel.recv_exit_status()
    if code: raise RuntimeError(f'Remote exit {code}: {stdout}\n{stderr}')
    return stdout + stderr


def progress():
    started = time.monotonic()
    last = [started]
    def report(done, total):
        now = time.monotonic()
        if now-last[0] >= 30 or done == total:
            print(json.dumps({'bytes':done, 'total':total,
                              'MB_per_second':done/1e6/max(now-started, .001)}), flush=True)
            last[0] = now
    return report
