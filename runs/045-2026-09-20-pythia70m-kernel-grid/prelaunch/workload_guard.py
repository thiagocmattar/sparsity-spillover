"""End the isolated workload at deadline without handling API credentials.

The independent workstation guard performs provider stop and billing cutoff.
This guard alone only ends compute processes; an idle Pod remains billable.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys
import time


def main():
    settings_path=Path(sys.argv[1])
    settings=json.loads(settings_path.read_text())
    settings_path.unlink()
    pod_id=settings['pod_id'];name=settings['name']
    if not (name.startswith('run045-') and pod_id.isalnum()):
        raise ValueError('Unscoped target')
    control=Path(settings['control'])
    if control != Path('/workspace/run045-control'):
        raise ValueError('Unscoped process directory')
    print(json.dumps({'event':'armed','pod_id':pod_id,
        'kind':'workload-only; provider stop is workstation-managed',
        'deadline_epoch':settings['deadline_epoch'],
        'utc':datetime.now(timezone.utc).isoformat()}),flush=True)
    while time.time()<settings['deadline_epoch']:
        time.sleep(min(30,max(0,settings['deadline_epoch']-time.time())))
    pid_path=control/'pipeline.pid'
    if pid_path.exists():
        pid=int(pid_path.read_text())
        try:
            if os.getpgid(pid)!=pid:
                raise RuntimeError('Workload must have its own process group')
            os.killpg(pid,signal.SIGTERM)
            print('WORKLOAD_TERM_SENT',flush=True)
        except ProcessLookupError:
            print('WORKLOAD_ALREADY_EXITED',flush=True)


if __name__=='__main__':main()
