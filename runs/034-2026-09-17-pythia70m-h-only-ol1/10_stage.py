"""Stage immutable source and initialization; build the exact input cache on seed."""
import argparse
import hashlib
import importlib.util
from pathlib import Path
import shlex
import time

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('cloud034',HERE/'07_cloud.py')
cloud=importlib.util.module_from_spec(spec);spec.loader.exec_module(cloud)


def stage(label):
    with cloud.connect(label) as c:
        cloud.upload(c,HERE/'prelaunch/source.tar.gz','/workspace/run034-source.tar.gz')
        expected=hashlib.sha256((HERE/'prelaunch/source.tar.gz').read_bytes()).hexdigest()
        assert cloud.command(c,'sha256sum /workspace/run034-source.tar.gz').split()[0]==expected
        cloud.command(c,f'mkdir -p {cloud.REMOTE}; tar -xzf /workspace/run034-source.tar.gz -C {cloud.REMOTE}')
        cloud.command(c,f'setsid bash {cloud.REMOTE_RUN}/00_setup_remote.sh > {cloud.CONTROL}/setup.log 2>&1 < /dev/null & echo $! > {cloud.CONTROL}/setup.pid')
    print('Environment installing on '+label,flush=True)


def seed_inputs(label):
    files=[HERE/'prelaunch/initialization/pythia70m-seed1234.safetensors',
           HERE/'prelaunch/initialization/pythia70m-seed1234-rng.pt',
           cloud.ROOT/'data/tokenized/minipile-pythia-14m-full/validation/tokens.int32.bin']
    with cloud.connect(label) as c:
        for p in files:
            remote=cloud.REMOTE+'/'+p.relative_to(cloud.ROOT).as_posix()
            staging='/opt/run034-inputs/'+p.name
            cloud.upload(c,p,staging)
            with p.open('rb') as h: expected=hashlib.file_digest(h,'sha256').hexdigest()
            assert cloud.command(c,'sha256sum '+shlex.quote(staging)).split()[0]==expected
            cloud.command(c,'cp '+shlex.quote(staging)+' '+shlex.quote(remote),timeout=120)
            assert cloud.command(c,'sha256sum '+shlex.quote(remote)).split()[0]==expected
        cloud.command(c,'touch '+cloud.CONTROL+'/initialization-ready')


def start_cache(label):
    script=f'''#!/bin/bash
set -euo pipefail
while [ ! -f {cloud.CONTROL}/environment-ready ]; do sleep 10; done
export HF_HOME=/opt/run034-hf-cache
export TOKENIZERS_PARALLELISM=true
export RAYON_NUM_THREADS=16
export PYTHONPATH={cloud.REMOTE}/src
cd {cloud.REMOTE}
/opt/run034-venv/bin/python - <<'PY'
import importlib.util,pathlib,sys,shutil,hashlib
path=pathlib.Path('runs/004-2026-08-29-pythia14m-full-pass-l1n/06_build_cache_from_hf.py')
s=importlib.util.spec_from_file_location('exact_cache',path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.CACHE_ROOT=pathlib.Path('/opt/run034-cache-build')
sys.argv=[str(path),'--splits','train','--batch-size','1024','--log-every-documents','50000']
m.main()
source=m.CACHE_ROOT/'train/tokens.int32.bin'
target=pathlib.Path('data/tokenized/minipile-pythia-14m-full/train/tokens.int32.bin')
assert not target.exists()
import subprocess
subprocess.run(['cp',str(source),str(target)],check=True)
with target.open('rb') as h:assert hashlib.file_digest(h,'sha256').hexdigest()=='da82a2ea2e0080c7fd681c7a93b07d3d9ff3d5357a8640895a82d536a1eaf97c'
print('Verified canonical training cache is ready',flush=True)
PY
touch {cloud.CONTROL}/cache-ready
'''
    with cloud.connect(label) as c:
        with c.open_sftp() as s:
            with s.open(cloud.CONTROL+'/build-cache.sh','w') as h:h.write(script)
        cloud.command(c,f'setsid bash {cloud.CONTROL}/build-cache.sh > {cloud.CONTROL}/cache.log 2>&1 < /dev/null & echo $! > {cloud.CONTROL}/cache.pid')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','inputs','cache']);p.add_argument('--label',required=True);a=p.parse_args()
    {'stage':stage,'inputs':seed_inputs,'cache':start_cache}[a.action](a.label)
