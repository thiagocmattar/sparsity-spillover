"""Retain read-only host details; never print credentials or environment secrets."""
import remote

COMMAND="""python3 - <<'PY'
import json,subprocess,time
from pathlib import Path
r=Path('/workspace/run037')
row={'epoch':time.time()}
commands={'cpu':['lscpu'],'gpu':['nvidia-smi','--query-gpu=name,uuid,driver_version,pstate,power.limit,clocks.max.sm,clocks.max.memory','--format=csv'],
          'storage':['df','-T','/workspace','/tmp']}
for key,cmd in commands.items():
 p=subprocess.run(cmd,capture_output=True,text=True);row[key]={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
p=r/'provenance/host-001.json'
with p.open('x') as f:json.dump(row,f,indent=2);f.write('\\n')
print('Recorded host-001.json')
PY"""

if __name__=='__main__':
    client=remote.connect()
    try:print(remote.execute(client,COMMAND))
    finally:client.close()
