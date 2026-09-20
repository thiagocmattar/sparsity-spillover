from pathlib import Path
import importlib.util,shlex
run=Path(__file__).resolve().parent.parent
spec=importlib.util.spec_from_file_location('remote',run/'10_remote.py');remote=importlib.util.module_from_spec(spec);spec.loader.exec_module(remote)
client,lease=remote.connect('latency');dest='/workspace/run041-latency'
try:
 remote.execute(client,f'mkdir -p {dest}/runtime {dest}/provenance')
 with client.open_sftp() as sftp:
  sftp.put(str(run/'latency/04_setup.sh'),dest+'/04_setup.sh')
  sftp.put(str(run/'latency/provenance/pip-freeze.txt'),dest+'/provenance/pip-freeze.txt')
 wrapper=f'bash {dest}/04_setup.sh --environment-only; code=$?; printf "%s\\n" "$code" > {dest}/runtime/environment-setup.exit'
 remote.execute(client,f'nohup setsid bash -c {shlex.quote(wrapper)} > {dest}/runtime/environment-setup.log 2>&1 < /dev/null & echo $! > {dest}/runtime/environment-setup.pid')
 print('Pinned latency environment installing while training continues',flush=True)
finally:client.close()
