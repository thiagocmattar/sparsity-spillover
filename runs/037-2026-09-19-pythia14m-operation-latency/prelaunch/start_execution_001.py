"""Start the approved qualification/measurement sequence after exact setup checks."""
import io
import json
from pathlib import Path
import remote

RUN=Path(__file__).resolve().parent.parent
lease=json.loads((RUN/'prelaunch/lease-001.json').read_text())
c=remote.connect()
try:
    print(remote.execute(c,"""cd /workspace/run037
test "$(cat runtime/setup-001.exit)" = 0
runtime/venv/bin/python - <<'PY'
from pathlib import Path
def packages(path):
 return {line.split('==',1)[0].lower().replace('_','-'):line.split('==',1)[1]
         for line in Path(path).read_text().splitlines() if '==' in line}
assert packages('provenance/pip-freeze.txt')==packages('runtime/pip-freeze.txt')
print('Exact environment package versions verified')
PY
"""))
    worker=('''#!/usr/bin/env bash
cd /workspace/run037
bash 05_execute.sh '''+str(lease['scientific_deadline_epoch'])+''' > runtime/execution-001.log 2>&1
code=$?
printf '%s\\n' "$code" > runtime/execution-001.exit
''').encode()
    s=c.open_sftp()
    s.putfo(io.BytesIO(worker),'/workspace/run037/runtime/execution-worker-001.sh')
    print(remote.execute(c,"""cd /workspace/run037
test ! -e runtime/execution-001.pid || exit 2
setsid bash runtime/execution-worker-001.sh < /dev/null > runtime/execution-launch-001.log 2>&1 &
printf '%s\\n' "$!" > runtime/execution-001.pid
cat runtime/execution-001.pid
"""))
    s.close()
finally: c.close()
