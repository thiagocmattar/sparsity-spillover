"""Arm the existing on-Pod guard using a transient private credential file."""
import argparse
import importlib.util
import json
from pathlib import Path
import time
import tomllib

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('remote032',HERE/'07_remote.py')
remote=importlib.util.module_from_spec(spec)
spec.loader.exec_module(remote)

def arm(pod,deadline,label='recovery2'):
    assert label in ('recovery2','recovery3')
    config=tomllib.loads((Path.home()/'.runpod/config.toml').read_text())
    settings={'pod_id':pod['id'],'name':pod['name'],'deadline_epoch':deadline,'api_key':config['apikey']}
    with remote.connect(pod['id']) as client:
        with client.open_sftp() as sftp:
            sftp.put(str(HERE/'13_recovery_guard.py'),f'/tmp/run032-{label}-guard.py')
            with sftp.open(f'/tmp/run032-{label}-auth.json','w') as handle:
                handle.chmod(0o600)
                handle.write(json.dumps(settings))
        remote.command(client,f'nohup python3 /tmp/run032-{label}-guard.py /tmp/run032-{label}-auth.json > /workspace/run032-{label}-guard.log 2>&1 < /dev/null &')
        for _ in range(12):
            value=remote.command(client,f'cat /workspace/run032-{label}-guard.log')
            if '"event": "armed"' in value:
                evidence=HERE/'prelaunch/cloud'/pod['id']
                (evidence/f'{label}-guard-armed.json').write_text(value,encoding='utf-8')
                if label=='recovery3':
                    remote.command(client,"pkill -f '^python3 /tmp/run032-recovery2-guard.py' || true")
                print(json.dumps({'pod':pod['id'],'guard':'armed','deadline_epoch':deadline}),flush=True)
                return
            time.sleep(1)
        raise RuntimeError('Recovery guard did not arm; stop this Pod')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--pod',required=True)
    parser.add_argument('--deadline',required=True,type=int)
    parser.add_argument('--label',choices=['recovery2','recovery3'],default='recovery2')
    args=parser.parse_args()
    pods=json.loads((HERE/'prelaunch/pods.json').read_text())
    arm(next(p for p in pods if p['id']==args.pod),args.deadline,args.label)
