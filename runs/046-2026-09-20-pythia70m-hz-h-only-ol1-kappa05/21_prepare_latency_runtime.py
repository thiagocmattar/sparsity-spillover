"""Overlap runtime installation and unchanged kernel compilation with training."""
import importlib.util
import json
from pathlib import Path
import shlex
import time

RUN=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('run046_remote',RUN/'10_remote.py')
remote=importlib.util.module_from_spec(spec);spec.loader.exec_module(remote)
CONTROL='/workspace/run046-control'
DEST='/workspace/run046-latency'


def main():
    client,lease=remote.connect('latency');started=time.monotonic()
    try:
        remote.execute(client,f'test ! -e {DEST}; mkdir -p {CONTROL} {DEST}/provenance')
        with client.open_sftp() as sftp:
            receipt=json.loads((RUN/'prelaunch/latency-bootstrap-001.receipt.json').read_text())
            sftp.put(str(RUN/receipt['path']),'/workspace/run046-latency-bootstrap.tar.gz')
            actual=remote.execute(client,'sha256sum /workspace/run046-latency-bootstrap.tar.gz').split()[0]
            assert actual==receipt['sha256']
            remote.execute(client,f'tar -xzf /workspace/run046-latency-bootstrap.tar.gz -C {DEST}')
            sftp.put(str(RUN/'12_deadline_guard.py'),CONTROL+'/12_deadline_guard.py')
            settings=CONTROL+'/environment-guard-settings.json'
            with sftp.open(settings,'w') as handle:
                handle.write(json.dumps(dict(pod_id=lease['pod']['id'],name=lease['pod']['name'],
                    deadline_epoch=lease['deadline_epoch'],control=CONTROL)))
        remote.execute(client,f'nohup python3 -u {CONTROL}/12_deadline_guard.py {settings} > {CONTROL}/environment-guard.log 2>&1 < /dev/null &')
        prepare=f'''set -euo pipefail
bash {DEST}/04_setup.sh --environment-only
export PATH={DEST}/runtime/venv/bin:/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda TORCH_EXTENSIONS_DIR={DEST}/runtime/extensions TRITON_CACHE_DIR={DEST}/runtime/triton MAX_JOBS=2 HF_HUB_OFFLINE=1
unset CUBLAS_WORKSPACE_CONFIG
python -u {DEST}/00_precompile.py
'''
        with client.open_sftp() as sftp:
            with sftp.open(CONTROL+'/latency-prepare.sh','w') as handle:handle.write(prepare)
        wrapper=f'bash {CONTROL}/latency-prepare.sh; code=$?; printf "%s\\n" "$code" > {CONTROL}/environment-setup.exit'
        remote.execute(client,f'nohup setsid bash -c {shlex.quote(wrapper)} > {CONTROL}/environment-setup.log 2>&1 < /dev/null & echo $! > {CONTROL}/pipeline.pid')
        record=dict(pod_id=lease['pod']['id'],role='latency',seconds=time.monotonic()-started,
            action='detached runtime setup and unchanged kernel compilation',scientific_checkpoint_inputs_uploaded=False,
            bootstrap_sha256_verified=actual)
        (RUN/'prelaunch/latency-environment-001.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record),flush=True)
    finally:client.close()


if __name__=='__main__':main()
