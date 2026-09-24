"""One-off Run054 cache distribution and independent GPU-worker startup."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import tarfile
import time

ROOT = Path('/workspace/sparsity-spillover')
RUN = ROOT/'runs/054-2026-09-24-pythia31m-t2-ph'
CONTROL = Path('/workspace/run054-control')
PYTHON = '/workspace/run054-venv/bin/python'
NODES = json.loads((CONTROL/'nodes.json').read_text())
DEADLINE = '2026-09-24T19:01:00Z'
LAST_START = datetime.fromisoformat(DEADLINE.replace('Z','+00:00')).timestamp()-3600


def emit(**fields):
    print(json.dumps(dict(utc=datetime.now().isoformat(),**fields)),flush=True)


def pipeline_command(node,index):
    tag = f'parallel-{index}-001'
    args = [PYTHON,str(RUN/'15_parallel_train.py'),'--workers',*node['conditions'],
            '--gpus',*[str(i) for i in range(len(node['conditions']))],
            '--deadline',DEADLINE,'--tag',tag]
    return ('test -f /workspace/run054-control/environment-ready && '+
        shlex.join([PYTHON,str(RUN/'12_check_inputs.py')])+' && '+
        'nohup setsid '+shlex.join(args)+
        ' > /workspace/run054-control/pipeline.log 2>&1 < /dev/null &')


def source_ready():
    upload = CONTROL/'input-data.tar'
    while time.time() < LAST_START:
        if (CONTROL/'cache-ready').exists():
            return 'pinned_hf_rebuild'
        if upload.exists() and upload.stat().st_size == 5969633280:
            with upload.open('rb') as handle:
                digest = hashlib.file_digest(handle,'sha256').hexdigest()
            if digest != '86415e9b8b3f32f4a0fbfb210d37fb4bd1fea66442a15b311c1f44ff2b2652b5':
                raise RuntimeError('Uploaded cache archive hash mismatch')
            found = subprocess.run(['pgrep','-f','^/workspace/run054-venv/bin/python runs/004-2026-08-29-pythia14m-full-pass-l1n/06_build_cache_from_hf.py'],capture_output=True,text=True)
            for pid in found.stdout.split():
                try: os.kill(int(pid),signal.SIGTERM)
                except ProcessLookupError: pass
            with tarfile.open(upload) as archive:
                archive.extractall(ROOT,filter='data')
            return 'existing_cache_upload'
        time.sleep(5)
    raise RuntimeError('Data preparation reached retrieval reserve')


def ssh(node):
    s = node['ssh']
    return ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','StrictHostKeyChecking=accept-new',
            '-i',str(CONTROL/'transfer-key'),'-p',str(s['port']),s['username']+'@'+s['host']]


def transfer_and_start(index):
    node = NODES[index]
    with (CONTROL/'cache-to-workers.tar').open('rb') as handle:
        subprocess.run(ssh(node)+['tar -xf - -C '+str(ROOT)],stdin=handle,check=True,timeout=900)
    # Source/cache hashes are checked before the detached process can execute.
    command = pipeline_command(node,index)
    result = subprocess.run(ssh(node)+[command],capture_output=True,text=True,check=True,timeout=120)
    emit(stage='worker_launched',pod=node['name'],conditions=node['conditions'],output=result.stdout)


def main():
    method = source_ready()
    spec = importlib.util.spec_from_file_location('cache_builder',ROOT/'runs/004-2026-08-29-pythia14m-full-pass-l1n/06_build_cache_from_hf.py')
    builder = importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    rows = []
    for split in ('train','validation'):
        path = ROOT/'data/tokenized/minipile-pythia-14m-full'/split
        builder.require_exact(json.loads((path/'metadata.json').read_text()),split,tokens_path=path/'tokens.int32.bin')
        for name in ('metadata.json','tokens.int32.bin'):
            item = path/name
            with item.open('rb') as handle:
                digest = hashlib.file_digest(handle,'sha256').hexdigest()
            rows.append(dict(path=item.relative_to(ROOT).as_posix(),bytes=item.stat().st_size,sha256=digest))
    (ROOT/'data-transfer.json').write_text(json.dumps(dict(files=rows,method=method),indent=2)+'\n')
    emit(stage='source_cache_verified',method=method)
    # Launch the two seed-node conditions while copies proceed to the other GPUs.
    subprocess.run(['bash','-lc',pipeline_command(NODES[1],1)],check=True)
    emit(stage='worker_launched',pod=NODES[1]['name'],conditions=NODES[1]['conditions'])
    with tarfile.open(CONTROL/'cache-to-workers.tar','w') as archive:
        for row in rows: archive.add(ROOT/row['path'],arcname=row['path'],recursive=False)
        archive.add(ROOT/'data-transfer.json',arcname='data-transfer.json',recursive=False)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(transfer_and_start,i) for i in (0,2,3)]
        for future in as_completed(futures): future.result()
    (CONTROL/'transfer-key').unlink()
    emit(stage='complete',private_transfer_key_removed=True)
    (CONTROL/'distribution-ready').touch()


if __name__ == '__main__':
    main()
