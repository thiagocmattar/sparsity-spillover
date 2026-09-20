from pathlib import Path
import importlib.util,json,shlex,time
run=Path(__file__).resolve().parent.parent
spec=importlib.util.spec_from_file_location('remote',run/'10_remote.py');remote=importlib.util.module_from_spec(spec);spec.loader.exec_module(remote)
client,lease=remote.connect('training');control='/workspace/run041-control';dest=remote.REMOTE+'/runs/'+run.name
try:
 assert remote.execute(client,f'cat {control}/pipeline.exit').strip()=='1'
 assert not remote.execute(client,f'cd {remote.REMOTE} && git status --porcelain').strip()
 receipt=json.loads((run/'prelaunch/source-003.receipt.json').read_text())
 with client.open_sftp() as sftp:
  sftp.put(str(run/receipt['path']),'/workspace/run041-source-003.bundle')
  assert remote.execute(client,'sha256sum /workspace/run041-source-003.bundle').split()[0]==receipt['sha256']
  remote.execute(client,f'cd {remote.REMOTE} && git fetch /workspace/run041-source-003.bundle HEAD && git checkout '+receipt['deployment_commit'])
  remote.execute(client,f'mkdir -p {dest}/prelaunch/attempt-001 {control}/attempt-001')
  remote.execute(client,f'mv {dest}/prelaunch/remote-preflight-*.json {dest}/prelaunch/attempt-001/; mv {control}/preflight-*.log {control}/pipeline.log {control}/pipeline.exit {control}/pipeline.pid {control}/attempt-001/')
  sftp.put(str(run/'inputs/random-initialization/model.safetensors'),dest+'/inputs/random-initialization/model.safetensors')
  actual=remote.execute(client,f'sha256sum {dest}/inputs/random-initialization/model.safetensors').split()[0]
  assert actual==json.loads((run/'inputs/random-initialization/provenance.json').read_text())['file_sha256']
  assert not remote.execute(client,f'cd {remote.REMOTE} && git status --porcelain').strip()
  wrapper=f'export RUN041_DEADLINE_EPOCH={lease["deadline_epoch"]}; bash {dest}/08_pipeline.sh; code=$?; printf "%s\\n" "$code" > {control}/pipeline.exit'
  remote.execute(client,f'nohup setsid bash -c {shlex.quote(wrapper)} > {control}/pipeline.log 2>&1 < /dev/null & echo $! > {control}/pipeline.pid')
  row={'deployment_commit':receipt['deployment_commit'],'source_sha256':receipt['sha256'],'initializer_file_sha256':actual,'preserved_failed_preflight':'attempt-001','deadline_unchanged':lease['deadline_utc']}
  (run/'prelaunch/retry-training-002.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row),flush=True)
finally:client.close()
