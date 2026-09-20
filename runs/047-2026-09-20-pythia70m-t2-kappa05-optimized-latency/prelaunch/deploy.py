"""Deploy only to an already approved, recorded Pod; never provision resources."""
import json
from pathlib import Path
import shlex
import subprocess
import time
import remote

RUN=Path(__file__).resolve().parent.parent
DEST='/workspace/run047'
CONTROL='/workspace/run047-control'


def main():
    lease=json.loads((RUN/'prelaunch/lease-001.json').read_text())
    assert lease['pod']['name']=='run047-70m-t2-kappa05-001'
    assert 0 < lease['deadline_epoch']-time.time() <= 90*60
    client=remote.connect(refresh=True)
    started=time.monotonic()
    try:
        remote.execute(client,f'mkdir -p {DEST}/provenance {CONTROL}')
        with client.open_sftp() as sftp:
            for source,target in [(RUN/'04_setup.sh',DEST+'/04_setup.sh'),
                    (RUN/'provenance/pip-freeze.txt',DEST+'/provenance/pip-freeze.txt'),
                    (RUN/'prelaunch/workload_guard.py',CONTROL+'/workload_guard.py')]:
                sftp.put(str(source),target)
            settings=CONTROL+'/guard-settings.json'
            with sftp.open(settings,'w') as stream:
                stream.write(json.dumps(dict(pod_id=lease['pod']['id'],name=lease['pod']['name'],
                    deadline_epoch=lease['deadline_epoch'],control=CONTROL)))
            remote.execute(client,f'nohup python3 -u {CONTROL}/workload_guard.py {settings} > {CONTROL}/guard.log 2>&1 < /dev/null &')
            command=f'bash {DEST}/04_setup.sh; code=$?; printf "%s\\n" "$code" > {CONTROL}/environment.exit'
            remote.execute(client,f'nohup setsid bash -c {shlex.quote(command)} > {CONTROL}/environment.log 2>&1 < /dev/null &')
        receipt=json.loads((RUN/'bundles/receipt-001.json').read_text())
        info=json.loads((RUN/'prelaunch/ssh-001.json').read_text())
        subprocess.run(['scp','-O','-q','-o','BatchMode=yes','-o',
            f'UserKnownHostsFile={RUN / "prelaunch/known_hosts"}', '-o','StrictHostKeyChecking=yes',
            '-o','ConnectTimeout=20','-i',info['ssh_key']['path'],'-P',str(info['port']),
            str(RUN/receipt['path']),f'root@{info["ip"]}:/workspace/run047-input-001.tar.gz'],
            check=True,timeout=max(1,lease['deadline_epoch']-time.time()-2400))
        script=f'''#!/usr/bin/env bash
set -euo pipefail
cd {DEST}
printf '%s  %s\\n' {receipt['sha256']} /workspace/run047-input-001.tar.gz | sha256sum -c -
tar -xzf /workspace/run047-input-001.tar.gz -C {DEST}
while test ! -f {CONTROL}/environment.exit; do sleep 10; done
test "$(cat {CONTROL}/environment.exit)" = 0
export RUN047_DEADLINE_EPOCH={lease['deadline_epoch']}
runtime/venv/bin/python 01_prepare.py verify
runtime/venv/bin/python 09_archive_cache.py
touch runtime/setup-complete
bash 05_execute.sh 03_execute.py --phase preflight --tag 001 --deadline-epoch {lease['deadline_epoch']}
bash 05_execute.sh 03_execute.py --phase final --tag 001 --deadline-epoch {lease['deadline_epoch']}
runtime/venv/bin/python 06_reduce.py --tag 001
runtime/venv/bin/python 08_collect.py --tag 001
'''
        (RUN/'prelaunch/pipeline-001.sh').write_text(script,encoding='utf-8',newline='\n')
        with client.open_sftp() as sftp:
            with sftp.open(CONTROL+'/pipeline-001.sh','w') as stream:stream.write(script)
        command=f'bash {CONTROL}/pipeline-001.sh; code=$?; printf "%s\\n" "$code" > {CONTROL}/pipeline.exit'
        remote.execute(client,f'nohup setsid bash -c {shlex.quote(command)} > {CONTROL}/pipeline.log 2>&1 < /dev/null & echo $! > {CONTROL}/pipeline.pid')
        result=dict(status='uploaded_pipeline_started',pod_id=lease['pod']['id'],
                    seconds=time.monotonic()-started,archive=receipt,deadline_epoch=lease['deadline_epoch'])
        (RUN/'prelaunch/upload-001.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
        print(json.dumps(result),flush=True)
    finally:client.close()


if __name__=='__main__':main()
