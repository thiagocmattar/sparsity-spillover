"""Upload the four endpoints and collect the scoped Run041 latency Pod outputs."""
import argparse
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import paramiko

RUN=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('run041_transport',RUN/'10_remote.py')
remote=importlib.util.module_from_spec(spec);spec.loader.exec_module(remote)
CONTROL='/workspace/run041-control'
DEST='/workspace/run041-latency'


def launch(client,lease):
    remote.execute(client,f'mkdir -p {CONTROL}; test ! -e {DEST}')
    sftp=paramiko.SFTPClient.from_transport(client.get_transport(),window_size=64*1024**2,max_packet_size=1024**2)
    try:
        sftp.put(str(RUN/'12_deadline_guard.py'),CONTROL+'/12_deadline_guard.py')
        settings=CONTROL+'/guard-settings.json'
        with sftp.open(settings,'w') as handle:
            handle.write(json.dumps({'pod_id':lease['pod']['id'],'name':lease['pod']['name'],
                'deadline_epoch':lease['deadline_epoch'],'control':CONTROL}))
        remote.execute(client,f'nohup python3 -u {CONTROL}/12_deadline_guard.py {settings} > {CONTROL}/guard.log 2>&1 < /dev/null &')
        receipt=json.loads((RUN/'bundles/latency-input-001.tar.receipt.json').read_text())
        started=time.monotonic();last=[started]
        def progress(done,total):
            now=time.monotonic()
            if now-last[0]>=30 or done==total:
                print(json.dumps({'upload_bytes':done,'total':total,
                    'MiB_per_second':done/1024**2/max(.01,now-started)}),flush=True)
                last[0]=now
        sftp.put(str(RUN/receipt['path']),'/workspace/run041-latency-input.tar.gz',callback=progress)
        digest=remote.execute(client,'sha256sum /workspace/run041-latency-input.tar.gz').split()[0]
        assert digest==receipt['sha256'],'Input archive transfer mismatch'
        remote.execute(client,f'mkdir {DEST}; tar -xzf /workspace/run041-latency-input.tar.gz -C {DEST}')
        guard=remote.execute(client,f'cat {CONTROL}/guard.log; test ! -e {settings}')
        assert '"event": "armed"' in guard
        pipeline=f'''set -euo pipefail
cd {DEST}
bash 04_setup.sh > {CONTROL}/setup.log 2>&1
bash 05_execute.sh smoke {lease['deadline_epoch']} 001 > {CONTROL}/smoke.log 2>&1
bash 05_execute.sh scientific {lease['deadline_epoch']} 001 > {CONTROL}/scientific.log 2>&1
runtime/venv/bin/python 08_collect.py > {CONTROL}/collection.log 2>&1
touch {CONTROL}/latency-sealed
'''
        with sftp.open(CONTROL+'/latency-pipeline.sh','w') as handle:handle.write(pipeline)
        wrapper=f'bash {CONTROL}/latency-pipeline.sh; code=$?; printf "%s\\n" "$code" > {CONTROL}/pipeline.exit'
        remote.execute(client,f'nohup setsid bash -c {shlex.quote(wrapper)} > {CONTROL}/pipeline.log 2>&1 < /dev/null & echo $! > {CONTROL}/pipeline.pid')
        (RUN/'prelaunch/upload-latency-001.json').write_text(json.dumps({'archive':receipt,
            'sha256_verified':True,'seconds':time.monotonic()-started,'pipeline':'detached',
            'workload_guard':'armed','provider_stop_guard':'local-process'},indent=2)+'\n')
        print('Four-endpoint archive verified; detached latency pipeline launched',flush=True)
    finally:sftp.close()


def retrieve(client):
    staging=RUN/'latency/retrieval';staging.mkdir(exist_ok=True)
    with client.open_sftp() as sftp:
        for name in ['receipt-001.json','output-001.tar.gz']:
            sftp.get(DEST+'/transfer/'+name,str(staging/name))
    subprocess.run([sys.executable,str(RUN/'latency/09_verify_retrieval.py')],check=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['launch','retrieve'])
    args=parser.parse_args();client,lease=remote.connect('latency')
    try:
        if args.action=='launch':launch(client,lease)
        else:retrieve(client)
    finally:client.close()
