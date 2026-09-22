"""Transfer hashed import-routing recovery and resume within the original deadline."""
import ast
import json
from pathlib import Path
import shlex
import sys
import remote

RUN=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(RUN))
from io_utils import read, write, record

folder=RUN/'provenance/recovery-001'
for path in folder.glob('*.py'):ast.parse(path.read_text(encoding='utf-8'))
manifest={'reason':'Archived diagnostics module shadowed the intended run-local module; no kernel/data/gate/numerical changes',
          'failed_attempt':record(RUN/'artifacts/attempts/final-c24-r1-001/result.json'),
          'source_freeze':record(RUN/'provenance/source-freeze.json'),
          'selection':record(RUN/'provenance/selection.json'),
          'files':[record(p) for p in sorted(folder.glob('*.py'))],
          'scope':'One full-length diagnostic/profile smoke, retry failed final as attempt002, then original remaining order. Failed qualification retained.'}
write(folder/'manifest.json',manifest)
deadline=read(RUN/'prelaunch/lease-001.json')['deadline_epoch']
client=remote.connect()
try:
    remote.execute(client,'test ! -f /workspace/run049/artifacts/controller.lock && mkdir -p /workspace/run049/provenance/recovery-001')
    with client.open_sftp() as sftp:
        for path in folder.iterdir():
            if path.is_file():sftp.put(str(path),'/workspace/run049/provenance/recovery-001/'+path.name)
    check="from io_utils import read,verify; [verify(p) for p in read('provenance/recovery-001/manifest.json')['files']]; print('Recovery hashes verified')"
    remote.execute(client,'cd /workspace/run049 && runtime/venv/bin/python -c '+shlex.quote(check))
    script=f'''set -euo pipefail
cd /workspace/run049
export RUN049_DEADLINE_EPOCH={deadline}
set +e
bash 05_execute.sh provenance/recovery-001/continue.py --deadline-epoch {deadline}
code=$?
if test "$code" = 0; then runtime/venv/bin/python provenance/recovery-001/reduce.py; code=$?; fi
runtime/venv/bin/python 08_collect.py --tag 002
exit "$code"
'''
    write(RUN/'prelaunch/recovery-launch-001.json',{'manifest':record(folder/'manifest.json'),'deadline_epoch':deadline})
    (RUN/'prelaunch/pipeline-002.sh').write_text(script,encoding='utf-8',newline='\n')
    with client.open_sftp() as sftp:
        with sftp.open('/workspace/run049-control/pipeline-002.sh','w') as stream:stream.write(script)
    command='bash /workspace/run049-control/pipeline-002.sh; code=$?; printf "%s\\n" "$code" > /workspace/run049-control/pipeline-002.exit'
    print(remote.execute(client,'nohup setsid bash -c '+shlex.quote(command)+' > /workspace/run049-control/pipeline-002.log 2>&1 < /dev/null & echo $! > /workspace/run049-control/pipeline.pid'),flush=True)
    print('RECOVERY_LAUNCHED',flush=True)
finally:client.close()
