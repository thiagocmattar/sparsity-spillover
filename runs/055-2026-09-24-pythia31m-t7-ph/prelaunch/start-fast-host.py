"""Verify the narrowly declared local cache alias, then detach the remaining pair."""
import hashlib,json,os,pathlib,subprocess,time
root=pathlib.Path('/workspace/sparsity-spillover');run=root/'runs/055-2026-09-24-pythia31m-t7-ph'
control=pathlib.Path('/workspace/run055-control');fast=pathlib.Path('/opt/run055-train-tokens.int32.bin')
allowed='data/tokenized/minipile-pythia-14m-full/train/tokens.int32.bin'
assert json.loads((control/'fast-cache.json').read_text())['status']=='verified'
verified=[]
for name in ('deployment.json','data-transfer.json'):
    manifest=json.loads((root/name).read_text())
    for row in manifest['files']:
        path=(root/row['path']).resolve()
        assert path.is_relative_to(root) or (row['path']==allowed and path==fast)
        with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        assert path.stat().st_size==row['bytes'] and digest==row['sha256'],row['path']
        verified.append(row)
(control/'fast-input-verification.json').write_text(json.dumps(dict(status='verified',files=verified),indent=2)+'\n')
assert not (run/'artifacts/pipeline/parallel-1-001').exists()
args=['/workspace/run055-venv/bin/python',str(run/'15_parallel_train.py'),'--workers','a7-h-ol1-kappa-0p05','a7-h-ol1-kappa-0p1','--gpus','0','1','--deadline','2026-09-24T21:43:56Z','--tag','parallel-1-001']
with (control/'pipeline.log').open('a') as log:
    child=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
        env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'))
receipt=dict(pid=child.pid,command=args,status='launched',unix_time=time.time())
(control/'fast-host-launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
