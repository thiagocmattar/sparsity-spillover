"""Native RunPod transfer retry; preserve partial SFTP archives separately."""
import argparse
import concurrent.futures
import importlib.util
import json
from pathlib import Path
import re
import shlex
import subprocess
import time

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('run032_retrieve',HERE/'12_retrieve_worker.py')
retriever=importlib.util.module_from_spec(spec)
spec.loader.exec_module(retriever)
remote=retriever.remote


def retrieve(pod):
    dest=HERE/'retrieval'/('croc-'+pod['id'])
    dest.mkdir(parents=True,exist_ok=True)
    with remote.connect(pod['id']) as client:
        if remote.command(client,'cat /workspace/run032-training.exit').strip()!='0':
            raise RuntimeError('Unverified remote attempt')
        remote.command(client,"nohup bash -c '/workspace/run032-runpodctl send /workspace/run032-results.tar > /workspace/run032-results-send.log 2>&1; echo $? > /workspace/run032-results-send.exit' >/dev/null 2>&1 </dev/null &")
        for _ in range(20):
            code=remote.command(client,'head -n 1 /workspace/run032-results-send.log').strip()
            if re.fullmatch(r'[A-Za-z0-9-]+',code):break
            time.sleep(1)
        else:raise RuntimeError('No transfer code')
    print(json.dumps({'pod':pod['id'],'transfer':'started'}),flush=True)
    with (dest/'receive.log').open('wb') as handle:
        subprocess.run([str(remote.CLI),'receive',code],cwd=dest,stdout=handle,stderr=subprocess.STDOUT,check=True)
    retriever.retrieve(pod,archive_path=dest/'run032-results.tar')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--pod',action='append',required=True)
    args=parser.parse_args()
    pods=json.loads((HERE/'prelaunch/pods.json').read_text())
    selected=[p for p in pods if p['id'] in args.pod]
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(retrieve,selected))
