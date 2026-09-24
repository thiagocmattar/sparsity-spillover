"""Verify immutable cache, distribute it, then run the authorized parallel cohort."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib,importlib.util,json,os,pathlib,shlex,subprocess,tarfile,time
ROOT=pathlib.Path('/workspace/sparsity-spillover')
RUN=ROOT/'runs/055-2026-09-24-pythia31m-t7-ph'
CONTROL=pathlib.Path('/workspace/run055-control')
PYTHON='/workspace/run055-venv/bin/python'
NODES=json.loads((CONTROL/'nodes.json').read_text())
DEADLINE=NODES[0]['deadline']
LAST_START=datetime.fromisoformat(DEADLINE.replace('Z','+00:00')).timestamp()-3600

def emit(**fields):
    print(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),**fields)),flush=True)

def ssh(node):
    s=node['ssh']
    return ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','StrictHostKeyChecking=accept-new','-i',str(CONTROL/'transfer-key'),'-p',str(s['port']),s['username']+'@'+s['host']]

def copy(index):
    started=time.time()
    with (CONTROL/'input-data.tar').open('rb') as f:
        subprocess.run(ssh(NODES[index])+['tar -xf - -C '+str(ROOT)],stdin=f,check=True,timeout=900)
    emit(stage='cache_copied',node=index,seconds=time.time()-started)

def command(index):
    node=NODES[index]
    args=[PYTHON,str(RUN/'15_parallel_train.py'),'--workers',*node['conditions'],'--gpus',*[str(i) for i in range(len(node['conditions']))],'--deadline',DEADLINE,'--tag',f'parallel-{index}-001']
    return ('test -f /workspace/run055-control/environment-ready && '+shlex.join([PYTHON,str(RUN/'12_check_inputs.py')])+' && nohup setsid '+shlex.join(args)+' > /workspace/run055-control/pipeline.log 2>&1 < /dev/null &')

def main():
    archive=CONTROL/'input-data.tar'
    while time.time()<LAST_START:
        if archive.exists() and archive.stat().st_size==5969633280 and (CONTROL/'environment-ready').exists():break
        time.sleep(5)
    else:raise TimeoutError('Retrieval reserve reached')
    with archive.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    assert digest=='86415e9b8b3f32f4a0fbfb210d37fb4bd1fea66442a15b311c1f44ff2b2652b5'
    with tarfile.open(archive) as a:a.extractall(ROOT,filter='data')
    subprocess.run([PYTHON,str(RUN/'12_check_inputs.py')],check=True)
    emit(stage='source_cache_verified')
    # Exercise final reload, publication and complete diagnostic lifecycle while peer copies run.
    with ThreadPoolExecutor(max_workers=2) as pool:
        copies=[pool.submit(copy,i) for i in (1,2)]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        with (CONTROL/'lifecycle.log').open('x') as log:
            subprocess.run([PYTHON,str(RUN/'14_lifecycle_smoke.py'),'--attempt','cloud-001'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200)
        emit(stage='lifecycle_passed')
        for f in copies:f.result()
    for i in range(3):
        if i==0:subprocess.run(['bash','-lc',command(i)],check=True)
        else:subprocess.run(ssh(NODES[i])+[command(i)],capture_output=True,text=True,check=True,timeout=120)
        emit(stage='pipeline_started',node=i,conditions=NODES[i]['conditions'])
    (CONTROL/'transfer-key').unlink()
    (CONTROL/'distribution-ready').touch()
    emit(stage='complete',private_transfer_key_removed=True)

if __name__=='__main__':main()
