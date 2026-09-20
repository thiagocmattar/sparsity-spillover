"""Install the frozen RTX5090 runtime while the training Pod finishes."""
import importlib.util
import json
import socket
from pathlib import Path
import shlex

RUN=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('run044_transport',RUN/'10_remote.py')
remote=importlib.util.module_from_spec(spec);spec.loader.exec_module(remote)
CONTROL='/workspace/run044-control'
DEST='/workspace/run044-latency'

def main():
    client,lease=remote.connect('latency')
    try:
        remote.execute(client,f'mkdir -p {CONTROL} {DEST}/provenance')
        with client.open_sftp() as sftp:
            for source,target in [(RUN/'12_deadline_guard.py',CONTROL+'/12_deadline_guard.py'),
                (RUN/'latency/04_setup.sh',DEST+'/04_setup.sh'),
                (RUN/'latency/provenance/pip-freeze.txt',DEST+'/provenance/pip-freeze.txt')]:
                sftp.put(str(source),target)
            settings=CONTROL+'/guard-settings.json'
            with sftp.open(settings,'w') as handle:
                handle.write(json.dumps({'pod_id':lease['pod']['id'],'name':lease['pod']['name'],
                    'deadline_epoch':lease['deadline_epoch'],'control':CONTROL}))
        receipt=json.loads((RUN/'prelaunch/latency-bootstrap-001.receipt.json').read_text())
        transport=client.get_transport()
        transport.sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
        transport.sock.setsockopt(socket.SOL_SOCKET,socket.SO_SNDBUF,4*1024**2)
        channel=transport.open_session(window_size=64*1024**2,max_packet_size=1024**2)
        channel.settimeout(120);channel.exec_command('cat > /workspace/run044-latency-bootstrap.tar.gz')
        with (RUN/receipt['path']).open('rb') as handle:
            while chunk:=handle.read(1024**2):channel.sendall(chunk)
        channel.shutdown_write();assert channel.recv_exit_status()==0;channel.close()
        assert remote.execute(client,'sha256sum /workspace/run044-latency-bootstrap.tar.gz').split()[0]==receipt['sha256']
        remote.execute(client,f'tar -xzf /workspace/run044-latency-bootstrap.tar.gz -C {DEST}')
        remote.execute(client,f'nohup python3 -u {CONTROL}/12_deadline_guard.py {settings} > {CONTROL}/guard.log 2>&1 < /dev/null &')
        command=f'export PATH={DEST}/runtime/venv/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda TORCH_EXTENSIONS_DIR={DEST}/runtime/extensions TRITON_CACHE_DIR={DEST}/runtime/triton MAX_JOBS=2 HF_HUB_OFFLINE=1; unset CUBLAS_WORKSPACE_CONFIG; bash {DEST}/04_setup.sh --environment-only && {DEST}/runtime/venv/bin/python {DEST}/00_precompile.py; code=$?; printf "%s\\n" "$code" > {CONTROL}/environment.exit'
        remote.execute(client,f'nohup setsid bash -c {shlex.quote(command)} > {CONTROL}/setup.log 2>&1 < /dev/null & echo $! > {CONTROL}/pipeline.pid')
        print('Frozen latency environment setup launched with the scoped deadline guard',flush=True)
    finally:client.close()

if __name__=='__main__':main()
