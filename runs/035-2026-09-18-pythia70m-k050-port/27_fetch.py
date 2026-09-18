"""Retrieve the immutable output archive over the same encrypted relay."""
import json
from pathlib import Path
import re
import secrets
import subprocess
import time
from remote_io import HERE,ROOT,connect,execute


def main():
    staging=HERE/'retrieval';staging.mkdir(exist_ok=True)
    c=connect();s=c.open_sftp()
    s.get('/workspace/run035/transfer/receipt-001.json',str(staging/'receipt-001.json'))
    receipt=json.loads((staging/'receipt-001.json').read_text())
    code='6734-'+secrets.token_hex(16)
    script=f'''#!/usr/bin/env bash
set -euo pipefail
cd /workspace/run035/transfer
../runtime/runpodctl-v2.14.0 send output-001.tar.gz --code {code}
'''
    remote='/workspace/run035/runtime/output-transfer-key.sh'
    with s.open(remote,'w') as f:f.write(script)
    s.chmod(remote,0o700)
    execute(c,"cd /workspace/run035; nohup bash runtime/output-transfer-key.sh > runtime/output-transfer.log 2>&1 < /dev/null & echo $!")
    until=time.monotonic()+30
    while time.monotonic()<until:
        text=execute(c,'cat /workspace/run035/runtime/output-transfer.log')
        match=re.search(r'code is:\s*(\S+)',text)
        if match:code=match.group(1);break
        time.sleep(1)
    else:raise RuntimeError('Sender did not emit a transfer code')
    s.close();c.close()
    with (staging/'transfer.log').open('w',encoding='utf-8') as log:
        child=subprocess.Popen([str(ROOT/'tmp/runpodctl-v2.14.0.exe'),'receive',code],cwd=staging,stdout=log,stderr=log)
    print(json.dumps({'receiver_pid':child.pid,'bytes':receipt['bytes'],'archive_sha256':receipt['sha256']}),flush=True)


if __name__=='__main__':main()
