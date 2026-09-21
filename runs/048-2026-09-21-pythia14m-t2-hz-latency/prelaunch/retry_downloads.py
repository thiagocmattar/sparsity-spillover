"""Switch only the remaining download transport; exact wheel hashes stay fixed."""
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
python3 /workspace/run048-control/download_verified_wheels.py
uv pip install --python /tmp/run048-local-runtime/venv/bin/python --no-deps /tmp/run048-wheels/*.whl
uv pip install --python /tmp/run048-local-runtime/venv/bin/python --index-url https://pypi.org/simple --find-links https://download.pytorch.org/whl/cu128/torch/ -r "$TASK_DIR/provenance/pip-freeze.txt"
ln -s /tmp/run048-local-runtime/venv "$TASK_DIR/runtime/venv"
mkdir -p /tmp/run048-local-runtime/extensions
ln -s /tmp/run048-local-runtime/extensions "$TASK_DIR/runtime/extensions"
uv pip freeze --python "$TASK_DIR/runtime/venv/bin/python" > "$TASK_DIR/runtime/pip-freeze.txt"
diff -u "$TASK_DIR/provenance/pip-freeze.txt" "$TASK_DIR/runtime/pip-freeze.txt"
nvcc --version > "$TASK_DIR/runtime/nvcc.txt"
nvidia-smi -q > "$TASK_DIR/runtime/nvidia-smi-initial.txt"
touch "$TASK_DIR/runtime/environment-ready"
'''
(RUN/'prelaunch/setup-004.sh').write_text(script,encoding='utf-8',newline='\n')
client = remote.connect()
try:
    print(remote.execute(client, '''python3 - <<'PY'
import os,pathlib,signal
matches=[]
for p in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
 try:
  parts=p.read_bytes().split(b'\\0')
  if len(parts)>3 and pathlib.Path(parts[0].decode()).name=='uv' and parts[1:3]==[b'pip',b'install'] and b'/tmp/run048-local-runtime/venv/bin/python' in parts:
   matches.append(int(p.parent.name))
 except (FileNotFoundError,PermissionError):pass
assert len(matches)==1,matches
group=os.getpgid(matches[0])
assert group>1 and group!=os.getpgrp()
os.killpg(group,signal.SIGTERM)
print('Stopped slow run-owned download process group',group)
PY'''))
    with client.open_sftp() as sftp:
        sftp.put(str(RUN/'prelaunch/download_verified_wheels.py'),CONTROL+'/download_verified_wheels.py')
        with sftp.open(CONTROL+'/setup-004.sh','w') as stream:
            stream.write(script)
    command=f'bash {CONTROL}/setup-004.sh; code=$?; printf "%s\\n" "$code" > {CONTROL}/environment.exit'
    remote.execute(client,f'nohup setsid bash -c {shlex.quote(command)} > {CONTROL}/environment-004.log 2>&1 < /dev/null &')
    print('Detached verified-download recovery004 started',flush=True)
finally:
    client.close()
