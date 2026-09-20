from pathlib import Path
import importlib.util,shlex
run=Path(__file__).resolve().parent.parent
spec=importlib.util.spec_from_file_location('remote',run/'10_remote.py');remote=importlib.util.module_from_spec(spec);spec.loader.exec_module(remote)
client,lease=remote.connect('training');control='/workspace/run041-control';dest=remote.REMOTE+'/runs/'+run.name
try:
 with client.open_sftp() as sftp:sftp.put(str(run/'15_seal_training.py'),control+'/15_seal_training.py')
 wrapper=f'while test ! -e {control}/pipeline.exit; do sleep 30; done; test "$(cat {control}/pipeline.exit)" = 0 && /workspace/run041-venv/bin/python -u {control}/15_seal_training.py {dest}; code=$?; printf "%s\\n" "$code" > {control}/collection.exit'
 remote.execute(client,f'nohup bash -c {shlex.quote(wrapper)} > {control}/collection.log 2>&1 < /dev/null & echo $! > {control}/collection.pid')
 print('Artifact sealing will start automatically after training verification')
finally:client.close()
