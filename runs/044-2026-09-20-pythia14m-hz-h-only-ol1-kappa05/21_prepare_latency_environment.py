"""Install the frozen RTX5090 runtime while the training Pod finishes."""
import importlib.util
import json
from pathlib import Path
import shlex

RUN=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('run044_transport',RUN/'10_remote.py')
remote=importlib.util.module_from_spec(spec);spec.loader.exec_module(remote)
CONTROL='/workspace/run044-control'
DEST='/workspace/run044-latency'

def main():
    client,lease=remote.connect('latency')
    try:
        remote.execute(client,f'mkdir -p {CONTROL} {DEST}/provenance')
        with client.open_sftp() as sftp:
            for source,target in [(RUN/'12_deadline_guard.py',CONTROL+'/12_deadline_guard.py'),
                (RUN/'latency/04_setup.sh',DEST+'/04_setup.sh'),
                (RUN/'latency/provenance/pip-freeze.txt',DEST+'/provenance/pip-freeze.txt')]:
                sftp.put(str(source),target)
            settings=CONTROL+'/guard-settings.json'
            with sftp.open(settings,'w') as handle:
                handle.write(json.dumps({'pod_id':lease['pod']['id'],'name':lease['pod']['name'],
                    'deadline_epoch':lease['deadline_epoch'],'control':CONTROL}))
        remote.execute(client,f'nohup python3 -u {CONTROL}/12_deadline_guard.py {settings} > {CONTROL}/guard.log 2>&1 < /dev/null &')
        command=f'bash {DEST}/04_setup.sh --environment-only; code=$?; printf "%s\\n" "$code" > {CONTROL}/environment.exit'
        remote.execute(client,f'nohup setsid bash -c {shlex.quote(command)} > {CONTROL}/setup.log 2>&1 < /dev/null & echo $! > {CONTROL}/pipeline.pid')
        print('Frozen latency environment setup launched with the scoped deadline guard',flush=True)
    finally:client.close()

if __name__=='__main__':main()
