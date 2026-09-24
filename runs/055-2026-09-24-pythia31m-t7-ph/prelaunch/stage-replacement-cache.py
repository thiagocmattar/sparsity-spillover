"""Transfer the unchanged compressed cache to the two replacement H200s."""
from concurrent.futures import ThreadPoolExecutor
import json,pathlib,subprocess,time
run=pathlib.Path(__file__).resolve().parent.parent
nodes=json.loads((run/'prelaunch/parallel-connections-001.json').read_text())
receipt=json.loads((run/'prelaunch/compressed-cache.json').read_text())
key=str(pathlib.Path.home()/'.runpod/ssh/runpodctl-ssh-key')
def work(index):
    node=nodes[index];s=node['ssh'];started=time.time()
    ssh=['ssh','-i',key,'-p',str(s['port']),'-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3',s['username']+'@'+s['host']]
    with (run/'prelaunch/input-data.tar.gz').open('rb') as f:
        child=subprocess.Popen(ssh+['cat > /workspace/run055-control/input-data.tar.gz'],stdin=subprocess.PIPE)
        try:
            while block:=f.read(8*1024*1024):child.stdin.write(block)
            child.stdin.close()
            assert child.wait(timeout=600)==0
        finally:
            if child.poll() is None:child.kill();child.wait()
    prepare='EXPECTED='+repr(receipt)+'\nWORKERS='+repr(node['conditions'])+'\nTAG='+repr(f'parallel-{index}-002')+'\n'+'''
import hashlib,json,pathlib,subprocess,tarfile,time,os
c=pathlib.Path('/workspace/run055-control');root=pathlib.Path('/workspace/sparsity-spillover')
r=root/'runs/055-2026-09-24-pythia31m-t7-ph';archive=c/'input-data.tar.gz'
with archive.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
assert archive.stat().st_size==EXPECTED['bytes'] and digest==EXPECTED['sha256']
with tarfile.open(archive,'r:gz') as f:f.extractall(root,filter='data')
deadline=1790286236
while not (c/'environment-ready').exists():
 assert time.time()<deadline-7200,'Setup exceeded training reserve'
 time.sleep(20)
python='/workspace/run055-venv/bin/python'
subprocess.run([python,str(r/'12_check_inputs.py')],check=True)
args=[python,str(r/'15_parallel_train.py'),'--workers',*WORKERS,'--gpus','0','--deadline','2026-09-24T21:43:56Z','--tag',TAG]
(c/'replacement-launch.json').write_text(json.dumps(dict(command=args,unix_time=time.time()),indent=2)+'\\n')
subprocess.run(args,check=True,env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'))
'''
    deploy='SCRIPT='+repr(prepare)+'\n'+'''
import pathlib,subprocess,json
c=pathlib.Path('/workspace/run055-control');script=c/'replacement-prepare.py';script.write_text(SCRIPT)
with (c/'replacement-pipeline.log').open('x') as f:
 child=subprocess.Popen(['python3',str(script)],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=child.pid,status='cache_uploaded_and_setup_detached')))
'''
    result=subprocess.run(ssh+['python3 -'],input=deploy,text=True,capture_output=True,check=True)
    result=dict(pod=node['id'],seconds=time.time()-started,remote=json.loads(result.stdout),cache=receipt)
    (run/'prelaunch'/f'replacement-cache-{index}-002.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(work,[3,4]))
