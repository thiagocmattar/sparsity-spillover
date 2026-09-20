"""Overlap the isolated latency runtime installation with the end of training."""
import importlib.util
import json
from pathlib import Path
import shlex
import time

RUN=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('run046_remote',RUN/'10_remote.py')
remote=importlib.util.module_from_spec(spec);spec.loader.exec_module(remote)
CONTROL='/workspace/run046-control'
DEST='/workspace/run046-latency'


def main():
    client,lease=remote.connect('latency');started=time.monotonic()
    try:
        remote.execute(client,f'test ! -e {DEST}; mkdir -p {CONTROL} {DEST}/provenance')
        with client.open_sftp() as sftp:
            sftp.put(str(RUN/'latency/04_setup.sh'),DEST+'/04_setup.sh')
            sftp.put(str(RUN/'latency/provenance/pip-freeze.txt'),DEST+'/provenance/pip-freeze.txt')
            sftp.put(str(RUN/'12_deadline_guard.py'),CONTROL+'/12_deadline_guard.py')
            settings=CONTROL+'/environment-guard-settings.json'
            with sftp.open(settings,'w') as handle:
                handle.write(json.dumps(dict(pod_id=lease['pod']['id'],name=lease['pod']['name'],
                    deadline_epoch=lease['deadline_epoch'],control=CONTROL)))
        remote.execute(client,f'nohup python3 -u {CONTROL}/12_deadline_guard.py {settings} > {CONTROL}/environment-guard.log 2>&1 < /dev/null &')
        wrapper=f'bash {DEST}/04_setup.sh --environment-only; code=$?; printf "%s\\n" "$code" > {CONTROL}/environment-setup.exit'
        remote.execute(client,f'nohup setsid bash -c {shlex.quote(wrapper)} > {CONTROL}/environment-setup.log 2>&1 < /dev/null & echo $! > {CONTROL}/pipeline.pid')
        record=dict(pod_id=lease['pod']['id'],role='latency',seconds=time.monotonic()-started,
            action='detached environment-only setup',scientific_checkpoint_inputs_uploaded=False)
        (RUN/'prelaunch/latency-environment-001.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record),flush=True)
    finally:client.close()


if __name__=='__main__':main()
