"""Infrastructure retry: detach setup without an SSH-inheriting shell."""
import json
from pathlib import Path
import remote

COMMAND = """python3 - <<'PY'
import json,subprocess,time
from pathlib import Path
r=Path('/workspace/run038')
assert not (r/'runtime/setup-001.log').exists(), 'Setup already started'
assert (r/'runtime/deadline-guard.log').exists()
assert not (r/'runtime/guard-settings-key.json').exists()
infra=r/'artifacts/infrastructure/002-setup-detach';infra.mkdir(parents=True,exist_ok=True)
record={'epoch':time.time(),'reason':'Initial SSH shell retained channel after arming guard; input hash verified, setup never started. Scientific bundle unchanged.'}
cmd='bash 04_setup.sh > runtime/setup-001.log 2>&1; code=$?; printf "%s\\n" "$code" > runtime/setup-001.exit'
p=subprocess.Popen(['bash','-c',cmd],cwd=r,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
record['setup_pid']=p.pid
(infra/'launch.json').write_text(json.dumps(record,indent=2)+'\\n')
print(json.dumps(record))
PY"""

if __name__=='__main__':
    c=remote.connect()
    try:
        result=remote.execute(c,COMMAND)
        print(result)
        (Path(__file__).parent/'setup-detach-002.json').write_text(result)
    finally:c.close()
