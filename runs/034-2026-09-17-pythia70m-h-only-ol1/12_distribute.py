"""Copy one verified input archive to the assigned owned Pods over SSH."""
import importlib.util
import json
from pathlib import Path
import shlex

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('cloud034',HERE/'07_cloud.py')
c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
targets=[label for label in json.loads((HERE/'prelaunch/assignments.json').read_text()) if label!='worker2']
with c.connect('worker2') as seed:
    c.command(seed,"test -f /opt/run034-transfer-key || ssh-keygen -q -t ed25519 -N '' -f /opt/run034-transfer-key")
    public=c.command(seed,'cat /opt/run034-transfer-key.pub').strip()
    for label in targets:
        with c.connect(label) as dest:
            c.command(dest,"mkdir -p /root/.ssh; chmod 700 /root/.ssh; printf '%s\\n' "+shlex.quote(public)+" >> /root/.ssh/authorized_keys; chmod 600 /root/.ssh/authorized_keys")
    files=['data/tokenized/minipile-pythia-14m-full/'+split+'/'+name
           for split in ['train','validation'] for name in ['metadata.json','tokens.int32.bin']]
    files+=['runs/'+HERE.name+'/prelaunch/initialization/'+name for name in
            ['metadata.json','pythia70m-seed1234.safetensors','pythia70m-seed1234-rng.pt']]
    script=f'''#!/bin/bash
set -euo pipefail
trap 'echo $? > {c.CONTROL}/distribution.exit' EXIT
while [ ! -f {c.CONTROL}/cache-ready ] || [ ! -f {c.CONTROL}/initialization-ready ]; do sleep 10; done
tar -b 2048 -cf /opt/run034-inputs.tar -C {c.REMOTE} {' '.join(files)}
sha256sum /opt/run034-inputs.tar > /opt/run034-inputs.sha256
'''
    for label in targets:
        info=json.loads((HERE/'prelaunch'/f'ssh-{label}.json').read_text())
        endpoint='root@'+info['ip'];port=int(info['port'])
        options=f'-i /opt/run034-transfer-key -o StrictHostKeyChecking=accept-new -o BatchMode=yes -o ConnectTimeout=20'
        verify=f'cd /opt && sha256sum -c run034-inputs.sha256 && tar -xf run034-inputs.tar -C {c.REMOTE} && touch {c.CONTROL}/initialization-ready {c.CONTROL}/cache-ready'
        job=f'scp -B {options} -P {port} /opt/run034-inputs.tar /opt/run034-inputs.sha256 {endpoint}:/opt/ && ssh {options} -p {port} {endpoint} '+shlex.quote(verify)
        script+=f'( {job}; echo $? > {c.CONTROL}/copy-{label}.exit ) > {c.CONTROL}/copy-{label}.log 2>&1 &\n'
    script+='wait\n'
    with seed.open_sftp() as ftp:
        with ftp.open(c.CONTROL+'/distribute.sh','w') as h:h.write(script)
    c.command(seed,f'setsid bash {c.CONTROL}/distribute.sh > {c.CONTROL}/distribution.log 2>&1 < /dev/null & echo $! > {c.CONTROL}/distribution.pid')
print('Seed distribution armed for '+', '.join(targets))
