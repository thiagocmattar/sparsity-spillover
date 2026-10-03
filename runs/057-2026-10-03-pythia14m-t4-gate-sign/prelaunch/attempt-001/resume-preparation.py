"""Infrastructure recovery after the completed checkout exceeded SSH timeout."""
import importlib.util
import json
from pathlib import Path
import shlex
import socket
import sys
import paramiko
run=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(run))
spec=importlib.util.spec_from_file_location('run057_transport',run/'10_remote.py')
remote=importlib.util.module_from_spec(spec);spec.loader.exec_module(remote)
client,lease=remote.connect()
try:
    root=remote.REMOTE;control='/workspace/run057-control';rr=f'{root}/runs/{run.name}'
    if remote.execute(client,f'cd {root} && git rev-parse HEAD').strip()!=lease['source_commit']:
        raise RuntimeError('Wrong source commit')
    transport=client.get_transport()
    transport.sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
    transport.sock.setsockopt(socket.SOL_SOCKET,socket.SO_SNDBUF,8*1024**2)
    with paramiko.SFTPClient.from_transport(transport,window_size=64*1024**2,max_packet_size=1024**2) as sftp:
        remote.upload(sftp,run/'inputs/random-initialization/model.safetensors',f'{rr}/inputs/random-initialization/model.safetensors')
        expected=json.loads((run/'inputs/random-initialization/provenance.json').read_text())['file_sha256']
        if remote.execute(client,f'sha256sum {rr}/inputs/random-initialization/model.safetensors').split()[0]!=expected:
            raise RuntimeError('Replay hash differs')
        with sftp.open(f'{root}/.git/info/exclude','a') as handle:
            handle.write(f'\nruns/{run.name}/prelaunch/\nruns/{run.name}/artifacts/\nruns/{run.name}/transfer/\n')
    # At ~4.3MiB/s the 6GB workstation upload projects ~22 minutes. Benchmark
    # the frozen pinned HF route using plentiful host CPUs instead, then require
    # both original cache hashes before preflight. Download/Arrow scratch stays
    # on the fast 40GB container disk; canonical int32 caches stay persistent.
    wrapper=(f'cd {root}; bash {rr}/00_setup_remote.sh || exit $?; '
        'export HF_HOME=/tmp/run057-hf TOKENIZERS_PARALLELISM=true RAYON_NUM_THREADS=32; '
        f'/workspace/run057-venv/bin/python -u {rr}/06_build_cache_from_hf.py --batch-size 1024 '
        f'> {control}/cache.log 2>&1 || exit $?; touch {control}/cache-ready; '
        f'bash {rr}/08_pipeline.sh; code=$?; printf "%s\\n" "$code" > {control}/pipeline.exit')
    remote.execute(client,f'nohup setsid bash -c {shlex.quote(wrapper)} > {control}/pipeline.log 2>&1 < /dev/null & echo $! > {control}/pipeline.pid')
    print('Replay/source verified; detached environment/cache/preflight/training pipeline started',flush=True)
finally:client.close()
