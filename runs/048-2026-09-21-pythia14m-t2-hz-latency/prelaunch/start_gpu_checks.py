"""Verify installed versions independently of pip's direct-wheel URL notation."""
import json
from pathlib import Path
import shlex
import remote

RUN = Path(__file__).resolve().parent.parent
CONTROL = '/workspace/run048-control'
lease = json.loads((RUN/'prelaunch/lease-001.json').read_text())
client = remote.connect()
try:
    print(remote.execute(client, '''/workspace/run048/runtime/venv/bin/python - <<'PY'
import importlib.metadata as m,json,pathlib
r=pathlib.Path('/workspace/run048')
norm=lambda name:name.lower().replace('_','-')
expected={norm(name):version for name,version in (line.split('==') for line in (r/'provenance/pip-freeze.txt').read_text().splitlines())}
actual={norm(d.metadata['Name']):d.version for d in m.distributions()}
assert actual==expected,{'missing_or_wrong':{n:(v,actual.get(n)) for n,v in expected.items() if actual.get(n)!=v},'extra':set(actual)-set(expected)}
record={'status':'verified','packages':actual,'package_count':len(actual),
 'note':'pip freeze uses direct-wheel URLs for four hash-verified wheels; all installed distribution versions equal the original pins.'}
(r/'provenance/runtime-version-verification.json').write_text(json.dumps(record,indent=2)+'\\n')
assert len(actual)==58
print('All58 installed distribution versions match the original pins')
PY
/usr/local/cuda/bin/nvcc --version > /workspace/run048/runtime/nvcc.txt
nvidia-smi -q > /workspace/run048/runtime/nvidia-smi-initial.txt
touch /workspace/run048/runtime/environment-ready'''))
    script = f'''#!/usr/bin/env bash
set -euo pipefail
cd /workspace/run048
export RUN048_DEADLINE_EPOCH={lease['deadline_epoch']}
runtime/venv/bin/python 01_prepare.py verify
runtime/venv/bin/python 09_archive_cache.py
touch runtime/setup-complete
bash 05_execute.sh 06_cuda_checks.py
bash 05_execute.sh 03_execute.py --phase smoke --tag 001 --deadline-epoch {lease['deadline_epoch']}
bash 05_execute.sh 03_execute.py --phase final --tag 001 --deadline-epoch {lease['deadline_epoch']}
runtime/venv/bin/python 07_reduce.py --tag 001
runtime/venv/bin/python 08_collect.py --tag 001
'''
    (RUN/'prelaunch/pipeline-003.sh').write_text(script,encoding='utf-8',newline='\n')
    print(remote.execute(client, '''python3 - <<'PY'
import pathlib
c=pathlib.Path('/workspace/run048-control');r=pathlib.Path('/workspace/run048')
assert (c/'pipeline.exit').read_text().strip()=='1'
assert not (r/'artifacts/controller.lock').exists()
assert not (r/'artifacts/attempts').exists()
for name in ['pipeline.log','pipeline.pid','pipeline.exit','environment.exit']:
 p=c/name
 if p.exists():p.rename(c/(name+'.before-version-check'))
(c/'environment.exit').write_text('0\\n')
print('Retained failed setup/pipeline logs; no scientific attempts existed')
PY'''))
    with client.open_sftp() as sftp:
        with sftp.open(CONTROL+'/pipeline-003.sh','w') as stream:
            stream.write(script)
    command=f'bash {CONTROL}/pipeline-003.sh; code=$?; printf "%s\\n" "$code" > {CONTROL}/pipeline.exit'
    remote.execute(client,f'nohup setsid bash -c {shlex.quote(command)} > {CONTROL}/pipeline.log 2>&1 < /dev/null & echo $! > {CONTROL}/pipeline.pid')
    print('Detached GPU checks and timing pipeline started',flush=True)
finally:
    client.close()
