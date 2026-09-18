"""Read-only chunk-identity progress for the repaired frozen input transfer."""
import hashlib
import json
import time
from remote_io import HERE, connect, execute

local=[]
with (HERE/'bundles/input-001.tar').open('rb') as f:
    while block:=f.read(8388608):local.append(hashlib.sha256(block).hexdigest())
client=connect()
command="""python3 - <<'PY'
import hashlib,json
rows=[]
with open('/workspace/run036/relay-bundle-001/input-001.tar','rb') as f:
 while block:=f.read(8388608):rows.append(hashlib.sha256(block).hexdigest())
print(json.dumps(rows))
PY"""
remote=json.loads(execute(client,command,timeout=60));client.close()
print(json.dumps({'correct_chunks':sum(a==b for a,b in zip(local,remote)),
    'total_chunks':len(local),'utc_epoch':time.time()}))
