"""Replace slow downloads with verified same-image CUDA files; no scientific edits."""
from pathlib import Path
import shlex
import remote

RUN = Path(__file__).resolve().parent.parent
CONTROL = '/workspace/run048-control'
script = '''#!/usr/bin/env bash
set -euo pipefail
TASK_DIR=/workspace/run048
export PATH=/tmp/run048-local-runtime/bootstrap/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
export UV_CACHE_DIR=/tmp/run048-uv-cache UV_HTTP_TIMEOUT=120
python3 /workspace/run048-control/reuse_image_packages.py
uv pip install --python /tmp/run048-local-runtime/venv/bin/python --index-url https://pypi.org/simple --find-links https://download.pytorch.org/whl/cu128/torch/ -r "$TASK_DIR/provenance/pip-freeze.txt"
ln -s /tmp/run048-local-runtime/venv "$TASK_DIR/runtime/venv"
mkdir -p /tmp/run048-local-runtime/extensions
ln -s /tmp/run048-local-runtime/extensions "$TASK_DIR/runtime/extensions"
uv pip freeze --python "$TASK_DIR/runtime/venv/bin/python" > "$TASK_DIR/runtime/pip-freeze.txt"
nvcc --version > "$TASK_DIR/runtime/nvcc.txt"
nvidia-smi -q > "$TASK_DIR/runtime/nvidia-smi-initial.txt"
touch "$TASK_DIR/runtime/environment-ready"
'''
(RUN/'prelaunch/setup-002.sh').write_text(script,encoding='utf-8',newline='\n')
client = remote.connect()
try:
    # This is the inspected run-owned installation process, not an arbitrary process.
    print(remote.execute(client, '''python3 - <<'PY'
import os,pathlib,signal
p=pathlib.Path('/proc/220/cmdline')
command=p.read_bytes().replace(b'\\0',b' ').decode()
assert 'uv pip install' in command and '/tmp/run048-local-runtime/venv/bin/python' in command
group=os.getpgid(220)
assert group>1 and group!=os.getpgrp()
os.killpg(group,signal.SIGTERM)
print('Stopped original run-owned environment group',group)
PY'''))
    with client.open_sftp() as sftp:
        sftp.put(str(RUN/'prelaunch/reuse_image_packages.py'),CONTROL+'/reuse_image_packages.py')
        with sftp.open(CONTROL+'/setup-002.sh','w') as stream:
            stream.write(script)
    command=f'bash {CONTROL}/setup-002.sh; code=$?; printf "%s\\n" "$code" > {CONTROL}/environment.exit'
    remote.execute(client,f'nohup setsid bash -c {shlex.quote(command)} > {CONTROL}/environment-002.log 2>&1 < /dev/null &')
    print('Detached environment retry002 started',flush=True)
finally:
    client.close()
