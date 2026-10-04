"""One bounded cold-build exception; never change the Pod deadline or timing code."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import time
from io_utils import RUN,read,write


def command(pid):
    return Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ')


def main():
    attempt=RUN/'artifacts/attempts/smoke-k0-r1-004'
    status=read(attempt/'status.json')
    if status['stage']!='capture' or (attempt/'timing.json').exists():
        raise RuntimeError('Exception is limited to cold capture before timing')
    parent=int((RUN/'artifacts/controller.lock').read_text())
    identity=command(parent)
    if b'03b_execute_hz.py' not in identity or b'--phase smoke' not in identity or b'004' not in identity:
        raise RuntimeError('Unexpected controller identity')
    children=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            ppid=int((p/'stat').read_text().split(') ')[1].split()[1])
            if ppid==parent and b'--attempt smoke-k0-r1-004' in command(int(p.name)):
                children.append(int(p.name))
        except (FileNotFoundError,ProcessLookupError):pass
    if len(children)!=1:raise RuntimeError('One owned cold-smoke child required')
    child=children[0]
    first=json.loads((attempt/'events.jsonl').read_text().splitlines()[0])
    deadline=datetime.fromisoformat(first['utc'].replace('Z','+00:00')).timestamp()+900
    if time.time()>=deadline:raise RuntimeError('Cold-build deadline already exceeded')
    receipt={'controller_pid':parent,'child_pid':child,'original_leaf_seconds':600,
        'cold_capture_maximum_seconds':900,'exception_started_utc':datetime.now(timezone.utc).isoformat(),
        'exception_deadline_epoch':deadline,'scientific_code_or_inputs_changed':False,
        'timing_started_at_exception':False,'pod_deadline_changed':False}
    os.kill(parent,signal.SIGSTOP)
    write(RUN/'artifacts/control/cold-smoke-allowance.json',receipt)
    try:
        while time.time()<deadline and not (attempt/'result.json').exists():
            if not Path(f'/proc/{child}').exists():break
            time.sleep(min(10,max(0,deadline-time.time())))
        if not (attempt/'result.json').exists():
            try:os.kill(child,signal.SIGTERM)
            except ProcessLookupError:pass
            receipt['outcome']='cold-build limit reached or child exited without result'
        else:receipt['outcome']='child result retained'
    finally:
        if Path(f'/proc/{parent}').exists() and command(parent)==identity:
            os.kill(parent,signal.SIGCONT)
        receipt['controller_resumed_utc']=datetime.now(timezone.utc).isoformat()
        write(RUN/'artifacts/control/cold-smoke-allowance.json',receipt)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
