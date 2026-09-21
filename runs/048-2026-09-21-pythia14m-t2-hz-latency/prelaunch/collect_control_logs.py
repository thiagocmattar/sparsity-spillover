"""Verify the terminal infrastructure scripts/logs in addition to scientific outputs."""
import hashlib
import json
from pathlib import Path
import remote

RUN = Path(__file__).resolve().parent.parent
destination = RUN/'prelaunch/remote-control'
destination.mkdir(exist_ok=True)
client = remote.connect()
try:
    records = json.loads(remote.execute(client, '''python3 - <<'PY'
import hashlib,json,pathlib
c=pathlib.Path('/workspace/run048-control')
assert (c/'pipeline.exit').read_text().strip()=='0'
assert not pathlib.Path('/workspace/run048/artifacts/controller.lock').exists()
rows=[]
for p in sorted(c.iterdir()):
 if p.is_file() and not p.is_symlink():
  data=p.read_bytes()
  rows.append({'name':p.name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
print(json.dumps(rows))
PY'''))
    with client.open_sftp() as sftp:
        for row in records:
            assert Path(row['name']).name == row['name']
            target = destination/row['name']
            sftp.get('/workspace/run048-control/'+row['name'], str(target))
            data = target.read_bytes()
            assert len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256']
    (RUN/'prelaunch/control-retrieval.json').write_text(json.dumps({
        'status':'verified','files':records},indent=2)+'\n',encoding='utf-8',newline='\n')
    print('Verified',len(records),'terminal infrastructure files',flush=True)
finally:
    client.close()
