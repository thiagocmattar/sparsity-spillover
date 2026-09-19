"""Start the authorized detached experiment after a successful pinned setup."""
import json
from pathlib import Path
import remote

if __name__=='__main__':
    lease=json.loads((Path(__file__).parent/'lease-001.json').read_text())
    client=remote.connect()
    try:
        deadline=lease['deadline_epoch']-600
        command="""python3 - <<'PY'
import json,subprocess
from pathlib import Path
r=Path('/workspace/run039')
assert (r/'runtime/setup-complete').is_file()
assert not (r/'runtime/execution-001.log').exists(), 'Execution already started'
cmd='bash 05_execute.sh DEADLINE > runtime/execution-001.log 2>&1; code=$?; printf "%s\\n" "$code" > runtime/execution-001.exit'
p=subprocess.Popen(['bash','-c',cmd],cwd=r,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
print(json.dumps({'execution_pid':p.pid,'deadline':DEADLINE}))
PY""".replace('DEADLINE',str(deadline))
        print(remote.execute(client,command))
        print('Detached execution started with ten-minute retrieval reserve',flush=True)
    finally:client.close()
