from pathlib import Path
import importlib.util,json,shlex,time,socket
run=Path(__file__).resolve().parent.parent
spec=importlib.util.spec_from_file_location('remote',run/'10_remote.py');remote=importlib.util.module_from_spec(spec);spec.loader.exec_module(remote)
client,lease=remote.connect('training');control='/workspace/run041-control';dest=remote.REMOTE+'/runs/'+run.name
try:
 transport=client.get_transport();transport.sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1);transport.sock.setsockopt(socket.SOL_SOCKET,socket.SO_SNDBUF,4*1024**2)
 path=dest+'/inputs/random-initialization/model.safetensors';offset=int(remote.execute(client,'stat -c %s '+path));source=run/'inputs/random-initialization/model.safetensors'
 assert 0<=offset<=source.stat().st_size
 channel=transport.open_session(window_size=64*1024**2,max_packet_size=1024**2);channel.settimeout(120);channel.exec_command('cat >> '+path)
 started=time.monotonic()
 with source.open('rb') as handle:
  handle.seek(offset)
  while chunk:=handle.read(1024**2):channel.sendall(chunk)
 channel.shutdown_write();assert channel.recv_exit_status()==0
 actual=remote.execute(client,'sha256sum '+path).split()[0]
 assert actual==json.loads((run/'inputs/random-initialization/provenance.json').read_text())['file_sha256']
 assert not remote.execute(client,f'cd {remote.REMOTE} && git status --porcelain').strip()
 wrapper=f'export RUN041_DEADLINE_EPOCH={lease["deadline_epoch"]}; bash {dest}/08_pipeline.sh; code=$?; printf "%s\\n" "$code" > {control}/pipeline.exit'
 remote.execute(client,f'nohup setsid bash -c {shlex.quote(wrapper)} > {control}/pipeline.log 2>&1 < /dev/null & echo $! > {control}/pipeline.pid')
 row={'deployment_commit':'b6cc11bf5ad725d9274963c82bb92f41a2359332','initializer_file_sha256':actual,'preserved_failed_preflight':'attempt-001','deadline_unchanged':lease['deadline_utc'],'stream_resumed_at':offset,'stream_seconds':time.monotonic()-started}
 (run/'prelaunch/retry-training-002.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row),flush=True)
finally:client.close()
