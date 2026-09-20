from pathlib import Path
import importlib.util,json
run=Path(__file__).resolve().parent.parent
spec=importlib.util.spec_from_file_location('remote',run/'10_remote.py');remote=importlib.util.module_from_spec(spec);spec.loader.exec_module(remote)
client,_=remote.connect('training');dest=remote.REMOTE+'/runs/'+run.name
try:
 with client.open_sftp() as sftp:
  for folder in ['prelaunch','prelaunch/attempt-001']:
   local=run/folder;local.mkdir(parents=True,exist_ok=True)
   for name in sftp.listdir(dest+'/'+folder):
    if name.startswith('remote-preflight-') and name.endswith('.json'):sftp.get(dest+'/'+folder+'/'+name,str(local/name))
 remote.execute(client,'nvidia-smi -q > /workspace/run041-control/nvidia-smi-training.txt')
 rows=[json.loads(p.read_text()) for p in (run/'prelaunch').glob('remote-preflight-*.json')]
 print(json.dumps({'retrieved_preflights':len(rows),'passed':sum(r['status']=='passed' for r in rows),'all_initializers_match':len({r['initial_parameter_sha256'] for r in rows})==1}))
finally:client.close()
