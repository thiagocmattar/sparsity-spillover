"""Bounded transport probe for the unchanged Run045 payload."""
import concurrent.futures
import json
from pathlib import Path
import subprocess
import time

RUN = Path(__file__).resolve().parent.parent
info = json.loads((RUN/'prelaunch/ssh-002.json').read_text())


def probe(destination):
    started = time.monotonic()
    command = ['scp', '-O', '-q', '-o', 'BatchMode=yes', '-o',
               'StrictHostKeyChecking=accept-new', '-o',
               f'UserKnownHostsFile={RUN / "prelaunch/known_hosts"}',
               '-o', 'ConnectTimeout=15', '-i', info['ssh_key']['path'],
               '-P', str(info['port']), str(RUN/'bundles/transport-sample.bin'),
               f'root@{info["ip"]}:{destination}']
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=35)
        return dict(destination=destination, seconds=time.monotonic()-started,
                    code=result.returncode, stderr=result.stderr)
    except subprocess.TimeoutExpired:
        return dict(destination=destination, seconds=time.monotonic()-started,
                    status='timeout')


with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    results = list(pool.map(probe, ['/tmp/run045-transport-sample.bin',
                                   '/workspace/run045-control/transport-sample.bin']))
(RUN/'prelaunch/transport-probe-002.json').write_text(json.dumps(results, indent=2)+'\n')
print(json.dumps(results), flush=True)
