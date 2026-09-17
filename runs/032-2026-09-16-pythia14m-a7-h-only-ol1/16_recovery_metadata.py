"""Capture recovery metadata independently of large checkpoint transfers."""
import concurrent.futures
import importlib.util
import json
from pathlib import Path
import shlex

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('remote',HERE/'07_remote.py')
remote=importlib.util.module_from_spec(spec)
spec.loader.exec_module(remote)

def capture(pod):
    dest=HERE/'prelaunch/cloud'/pod['id']
    dest.mkdir(parents=True,exist_ok=True)
    with remote.connect(pod['id']) as client:
        for name in ['run032-training.exit','run032-results.sha256','run032-recovery-guard.log']:
            value=remote.command(client,'cat /workspace/'+name)
            (dest/name).write_text(value,encoding='utf-8')
        code="""import pathlib,tarfile
p=pathlib.Path('/workspace/sparsity-spillover/runs/032-2026-09-16-pythia14m-a7-h-only-ol1')
with tarfile.open('/workspace/run032-metadata.tar.gz','w:gz') as t:
 for f in (p/'artifacts').rglob('*'):
  if f.is_file() and f.suffix in ('.json','.jsonl','.md','.csv'):t.add(f,arcname=str(f.relative_to(p)))
"""
        remote.command(client,'python3 -c '+shlex.quote(code))
        expected=remote.command(client,'sha256sum /workspace/run032-metadata.tar.gz').split()[0]
        with client.open_sftp() as sftp:
            sftp.get('/workspace/run032-metadata.tar.gz',str(dest/'metadata.tar.gz'))
        assert remote.sha(dest/'metadata.tar.gz')==expected
        (dest/'metadata-sha256.json').write_text(json.dumps({'sha256':expected,'pod':pod['id']})+'\n')
        print(json.dumps({'metadata_verified':pod['id']}),flush=True)

if __name__=='__main__':
    pods=json.loads((HERE/'prelaunch/pods.json').read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(capture,p):p['id'] for p in pods if p['id']!='8o1uyxgh6vb6ym'}
        for f in concurrent.futures.as_completed(futures):
            try:f.result()
            except Exception as e:print(json.dumps({'pod':futures[f],'error':str(e)}),flush=True)
