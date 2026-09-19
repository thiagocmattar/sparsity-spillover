"""Infrastructure retry: expose the pinned venv's Ninja executable on PATH."""
import io
import json
from pathlib import Path
import remote

RUN=Path(__file__).resolve().parent.parent
lease=json.loads((RUN/'prelaunch/lease-001.json').read_text())
folder='artifacts/infrastructure/002-venv-path'
worker=('''#!/usr/bin/env bash
cd /workspace/run037
export PATH=/workspace/run037/runtime/venv/bin:$PATH
bash 05_execute.sh '''+str(lease['scientific_deadline_epoch'])+''' > '''+folder+'''/execution.log 2>&1
code=$?
printf '%s\\n' "$code" > '''+folder+'''/execution.exit
''').encode()
local=RUN/'prelaunch/attempts/002-venv-path'
local.mkdir(parents=True,exist_ok=True)
with (local/'worker.sh').open('xb') as handle:handle.write(worker)
c=remote.connect()
try:
    print(remote.execute(c,"""cd /workspace/run037
test "$(cat runtime/execution-001.exit)" = 1
test -x runtime/venv/bin/ninja
runtime/venv/bin/ninja --version
mkdir -p artifacts/infrastructure/002-venv-path
"""))
    s=c.open_sftp()
    s.putfo(io.BytesIO(worker),'/workspace/run037/'+folder+'/worker.sh')
    command=('cd /workspace/run037\n'
             'test ! -e '+folder+'/worker.pid || exit 2\n'
             'setsid bash '+folder+'/worker.sh < /dev/null > '+folder+'/launch.log 2>&1 &\n'
             'printf \'%s\\n\' "$!" > '+folder+'/worker.pid\n'
             'cat '+folder+'/worker.pid\n')
    print(remote.execute(c,command));s.close()
finally:c.close()
