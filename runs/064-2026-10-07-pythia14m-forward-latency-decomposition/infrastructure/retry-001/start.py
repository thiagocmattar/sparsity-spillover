"""Start only the owned finished Pod's shape-adapted fresh tag002 cohort."""
import importlib.util
import json
from pathlib import Path
import shlex

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('run064_cloud',ROOT/'06_cloud.py')
cloud=importlib.util.module_from_spec(spec);spec.loader.exec_module(cloud)
client,lease=cloud.connect()
try:
    if cloud.execute(client,'cat /workspace/run064-control/pipeline.exit').strip()!='1':raise ValueError('Expected completed failed controller')
    folder=ROOT/'infrastructure/retry-001';remote=cloud.REMOTE+'/infrastructure/retry-001'
    cloud.execute(client,f'mkdir -p {remote}')
    sources=[]
    with client.open_sftp() as sftp:
        for name in ['profile.py','controller.py','deadline-guard.py']:
            path=folder/name;sftp.put(str(path),remote+'/'+name)
            actual=cloud.execute(client,'sha256sum '+remote+'/'+name).split()[0]
            if actual!=cloud.sha(path):raise ValueError('Retry source mismatch')
            sources.append({'path':path.relative_to(ROOT).as_posix(),'sha256':actual,'bytes':path.stat().st_size})
    command="""python3 - <<'PY'
from pathlib import Path
import os,signal
control=Path('/workspace/run064-control')
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:cmd=(p/'cmdline').read_bytes()
 except OSError:continue
 if cmd.startswith(b'python3'+bytes([0])+b'-u'+bytes([0])+b'/workspace/run064-control/deadline-guard.py'+bytes([0])):os.kill(int(p.name),signal.SIGTERM)
(control/'pipeline.exit').unlink()
PY"""
    cloud.execute(client,command)
    env='export RUN064_BASE=/opt/run064-fast/dependencies/run063 RUN063_ARCHIVE=/opt/run064-fast/dependencies/run063/archive/root TORCH_EXTENSIONS_DIR=/opt/run064-fast/extensions TRITON_CACHE_DIR=/opt/run064-fast/triton MAX_JOBS=2 HF_HUB_OFFLINE=1 CUDA_HOME=/usr/local/cuda; export PATH=/opt/run064-fast/venv/bin:/usr/local/cuda/bin:$PATH; unset CUBLAS_WORKSPACE_CONFIG; '
    wrapper=env+f'export RUN064_DEADLINE={lease["deadline_epoch"]}; python -u {remote}/controller.py; code=$?; printf "%s\\n" "$code" > {cloud.CONTROL}/pipeline.exit'
    cloud.execute(client,f'nohup setsid bash -c {shlex.quote(wrapper)} > {cloud.CONTROL}/pipeline.log 2>&1 < /dev/null & echo $! > {cloud.CONTROL}/pipeline.pid')
    cloud.execute(client,f'nohup python3 -u {remote}/deadline-guard.py > {cloud.CONTROL}/deadline-guard-002.log 2>&1 < /dev/null &')
    cloud.write(ROOT/'prelaunch/retry-001.json',{'diagnostic_shape_adapter':True,'production_sources_unchanged':True,'measurements_unchanged':True,'tag':'002','sources':sources,'same_gpu_and_original_deadline':True})
    print('Detached tag002 cohort started; failed001 preserved')
finally:client.close()
