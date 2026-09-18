"""Runpodctl encrypted relay transfer after a measured slow direct SSH path."""
import argparse
import json
from pathlib import Path
import secrets
import re
import subprocess
import time
from remote_io import HERE,ROOT,connect,execute

def main():
    p=argparse.ArgumentParser();p.add_argument('--probe',action='store_true');p.add_argument('--bootstrap',action='store_true');p.add_argument('--tag',default='001');p.add_argument('--resume-bundle',action='store_true');a=p.parse_args()
    assert a.tag.isalnum()
    tag=('probe' if a.probe else ('bootstrap' if a.bootstrap else 'bundle'))+'-'+a.tag
    directory='bundle-001' if a.resume_bundle else tag
    assert not a.resume_bundle or not (a.probe or a.bootstrap)
    source=HERE/('inputs/validation.int32.bin' if a.probe else (f'bundles/bootstrap-{a.tag}.tar' if a.bootstrap else 'bundles/input-001.tar'))
    code='6734-'+secrets.token_hex(16)
    local_log=HERE/f'prelaunch/relay-{tag}.log'
    with local_log.open('w',encoding='utf-8') as log:
        sender=subprocess.Popen([str(ROOT/'tmp/runpodctl-v2.14.0.exe'),'send',str(source),'--code',code],stdout=log,stderr=log)
    # CLI appends its selected relay identifier; receiver needs the emitted code.
    until=time.monotonic()+20
    while time.monotonic()<until:
        match=re.search(r'code is:\s*(\S+)',local_log.read_text(errors='replace'))
        if match:code=match.group(1);break
        if sender.poll() is not None:raise RuntimeError('Transfer sender exited before emitting a code')
        time.sleep(.25)
    else:raise RuntimeError('Transfer sender did not emit a code')
    c=connect();s=c.open_sftp()
    script=f'''#!/usr/bin/env bash
set -euo pipefail
mkdir -p /workspace/run035/relay-{directory}
cd /workspace/run035
if [ ! -x runtime/runpodctl-v2.14.0 ]; then
  curl --max-time 120 -fsSL https://github.com/runpod/runpodctl/releases/download/v2.14.0/runpodctl-linux-amd64 -o runtime/runpodctl-v2.14.0
  echo '2e0fd370a52a0fc7e43a6434a209348a4f6836fcdf1ad2b093b609e938138be9  runtime/runpodctl-v2.14.0' | sha256sum -c -
  chmod 700 runtime/runpodctl-v2.14.0
fi
runtime/runpodctl-v2.14.0 version
cd relay-{directory}
/workspace/run035/runtime/runpodctl-v2.14.0 receive {code}
'''
    remote=f'/workspace/run035/runtime/relay-{tag}-key.sh'
    with s.open(remote,'w') as f:f.write(script)
    s.chmod(remote,0o700)
    cmd=f"cd /workspace/run035; nohup bash -c 'bash runtime/relay-{tag}-key.sh; echo $? > runtime/relay-{tag}.exit' > runtime/relay-{tag}.log 2>&1 < /dev/null & echo $!"
    remote_pid=execute(c,cmd).strip()
    receipt={'tag':tag,'sender_pid':sender.pid,'remote_pid':remote_pid,'started_epoch':time.time(),'bytes':source.stat().st_size}
    (HERE/f'prelaunch/relay-{tag}.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True);s.close();c.close()

if __name__=='__main__':main()
