"""Check portable kernel components on the finished GPU; no timing evidence."""
import pathlib,subprocess,tarfile
run=pathlib.Path(__file__).resolve().parent.parent
archive=run/'prelaunch/kernel-source-only.tar.gz'
prefix='runs/'+run.name+'/'
with tarfile.open(run/'prelaunch/input-training.tar') as training:already_staged=set(training.getnames())
with tarfile.open(run/'prelaunch/input-source.tar') as source,tarfile.open(archive,'w:gz',compresslevel=1) as out:
    for row in source.getmembers():
        if row.name not in already_staged:
            out.addfile(row,source.extractfile(row) if row.isfile() else None)
ssh=['ssh','-i',str(pathlib.Path.home()/'.runpod/ssh/runpodctl-ssh-key'),'-o','BatchMode=yes','-p','27548','root@103.196.86.181']
with archive.open('rb') as f:
    child=subprocess.Popen(ssh+['tar -xzf - -C /workspace/sparsity-spillover'],stdin=subprocess.PIPE)
    while block:=f.read(8*1024*1024):child.stdin.write(block)
    child.stdin.close();assert child.wait(timeout=120)==0
code='''import pathlib,subprocess,json,os
c=pathlib.Path('/workspace/run055-control');r=pathlib.Path('/workspace/sparsity-spillover/runs/055-2026-09-24-pythia31m-t7-ph')
verified=[v for p in (r/'artifacts').glob('verification-*.json') for v in json.loads(p.read_text()).get('conditions',[]) if v['condition']['id']=='a7-h-ol1-kappa-0' and v['status']=='verified']
assert len(verified)==1
script=c/'h200-components-003.py'
script.write_text("""import subprocess,pathlib,json,os
c=pathlib.Path('/workspace/run055-control');r=pathlib.Path('/workspace/sparsity-spillover/runs/055-2026-09-24-pythia31m-t7-ph')
env=dict(os.environ,PATH='/workspace/run055-venv/bin:'+os.environ['PATH'],CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
p=subprocess.run(['/workspace/run055-venv/bin/python',str(r/'latency/02_component_check.py')],env=env)
result=dict(returncode=p.returncode,scientific_measurement=False,scope='58 portable component cases; final RTX5090 qualification remains required')
if p.returncode==0:result['components']=json.loads((r/'latency/component-check.json').read_text())
(r/'artifacts/preflight-h200-components-003.json').write_text(json.dumps(result,indent=2))
(c/'h200-components-exit-003.json').write_text(json.dumps(result))
""")
compile(script.read_text(),str(script),'exec')
with (c/'h200-components-003.log').open('x') as f:
 p=subprocess.Popen(['python3',str(script)],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=p.pid,scientific_measurement=False)))
'''
p=subprocess.run(ssh+['python3 -'],input=code,text=True,capture_output=True,check=True)
(run/'prelaunch/h200-component-launch-003.json').write_text(p.stdout)
print(p.stdout)
