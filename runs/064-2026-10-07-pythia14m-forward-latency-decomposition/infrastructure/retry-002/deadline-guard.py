"""Credential-free process-group deadline for this run's detached pipeline."""
import json
import os
from pathlib import Path
import signal
import time

control=Path('/workspace/run064-control')
settings=json.loads((control/'guard-settings.json').read_text())
if settings['pod_id']!='p79ch842a0hagf' or settings['name']!='run064-forward-decomposition-001':raise ValueError('Scope')
print('ARMED',settings['deadline_epoch'],flush=True)
while time.time()<settings['deadline_epoch']:time.sleep(min(30,settings['deadline_epoch']-time.time()))
if (control/'pipeline.exit').exists():
    print('Pipeline already ended',flush=True);raise SystemExit(0)
pid=int((control/'pipeline.pid').read_text())
cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ')
if b'/workspace/run064/infrastructure/retry-002/controller.py' not in cmd:raise ValueError('Owned pipeline identity changed')
if os.getpgid(pid)!=pid:raise ValueError('Owned pipeline is not its own process group')
os.killpg(pid,signal.SIGTERM)
print('Deadline signalled owned pipeline; persistent evidence retained',flush=True)
