"""Replace a stalled package install with the identical working virtual environment."""
import pathlib,subprocess
code=r'''
import pathlib,subprocess,json,time,os,signal
c=pathlib.Path('/workspace/run055-control')
ssh=['ssh','-i','/root/.ssh/run055-transfer-key','-o','BatchMode=yes','-o','ConnectTimeout=20','-p','13409','root@213.181.104.53']
prepare="""import pathlib,os,signal,json
c=pathlib.Path('/workspace/run055-control');r=pathlib.Path('/workspace/sparsity-spillover/runs/055-2026-09-24-pythia31m-t7-ph')
assert not (c/'environment-ready').exists(),'Installer already finished; leave it alone'
assert not list((r/'artifacts/attempts').glob('*'))
for pid,expected in [(191,'00_setup_remote.sh'),(365,'python3 -')]:
 p=pathlib.Path('/proc')/str(pid)/'cmdline'
 if p.exists():
  assert expected in p.read_bytes().replace(b'\\0',b' ').decode()
  os.killpg(pid,signal.SIGTERM)
v=pathlib.Path('/workspace/run055-venv');assert not pathlib.Path('/workspace/run055-venv-incomplete-001').exists()
v.rename('/workspace/run055-venv-incomplete-001')
(c/'environment-copy-start.json').write_text(json.dumps(dict(source_pod='rjtpxhwiek5tpe',reason='stalled package downloads; same pinned image and paths')))
"""
subprocess.run(ssh+['python3 -'],input=prepare,text=True,check=True,timeout=30)
archive=c/'environment-transfer.tar.gz'
with archive.open('wb') as out:
 tar=subprocess.Popen(['tar','--exclude=__pycache__','-C','/workspace','-cf','-','run055-venv'],stdout=subprocess.PIPE)
 gzip=subprocess.Popen(['gzip','-1','-c'],stdin=tar.stdout,stdout=subprocess.PIPE);tar.stdout.close()
 upload=subprocess.Popen(ssh+['cat > /workspace/run055-control/environment-transfer.tar.gz'],stdin=subprocess.PIPE)
 import hashlib
 h=hashlib.sha256();size=0
 try:
  while block:=gzip.stdout.read(8*1024*1024):
   out.write(block);h.update(block);size+=len(block);upload.stdin.write(block)
  upload.stdin.close();assert upload.wait(timeout=120)==0
  assert gzip.wait(timeout=30)==0 and tar.wait(timeout=30)==0
 finally:
  for child in (upload,gzip,tar):
   if child.poll() is None:child.kill();child.wait()
receipt=dict(bytes=size,sha256=h.hexdigest(),source_pod='rjtpxhwiek5tpe',target_pod='wvygmjw28xks6r')
(c/'environment-transfer.json').write_text(json.dumps(receipt,indent=2))
finish='EXPECTED='+repr(receipt)+'\n'+"""import pathlib,hashlib,subprocess,json,sys,os,time
c=pathlib.Path('/workspace/run055-control');a=c/'environment-transfer.tar.gz'
with a.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==EXPECTED['sha256']
assert a.stat().st_size==EXPECTED['bytes']
subprocess.run(['tar','-xzf',str(a),'-C','/workspace'],check=True)
python='/workspace/run055-venv/bin/python'
subprocess.run([python,'-c','import sys,torch,transformers; assert sys.version_info[:2]==(3,12); assert torch.__version__=="2.11.0+cu128"; assert torch.version.cuda=="12.8"; assert transformers.__version__=="5.12.1"'],check=True)
with (c/'pip-freeze.txt').open('w') as f:subprocess.run(['/workspace/run055-bootstrap/bin/uv','pip','freeze','--python',python],stdout=f,check=True)
(c/'environment-copy-verified.json').write_text(json.dumps(dict(status='verified',archive=EXPECTED),indent=2))
(c/'environment-ready').touch()
r=pathlib.Path('/workspace/sparsity-spillover/runs/055-2026-09-24-pythia31m-t7-ph')
subprocess.run([python,str(r/'12_check_inputs.py')],check=True)
args=[python,str(r/'15_parallel_train.py'),'--workers','a7-h-ol1-kappa-0p1','--gpus','0','--deadline','2026-09-24T21:43:56Z','--tag','parallel-4-002']
with (c/'replacement-pipeline.log').open('x') as f:
 p=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'))
(c/'replacement-launch.json').write_text(json.dumps(dict(pid=p.pid,command=args,unix_time=time.time()),indent=2))
print(json.dumps(dict(pid=p.pid,status='launched')))
"""
subprocess.run(ssh+['python3 -'],input=finish,text=True,check=True,timeout=900)
'''
deploy='SCRIPT='+repr(code)+'\n'+'''
import pathlib,subprocess,json
c=pathlib.Path('/workspace/run055-control');p=c/'relay-environment.py';p.write_text(SCRIPT)
with (c/'relay-environment.log').open('x') as f:
 child=subprocess.Popen(['python3',str(p)],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=child.pid,status='environment_relay_started')))
'''
ssh=['ssh','-i',str(pathlib.Path.home()/'.runpod/ssh/runpodctl-ssh-key'),'-o','BatchMode=yes','-p','27548','root@103.196.86.181']
p=subprocess.run(ssh+['python3 -'],input=deploy,text=True,capture_output=True,check=True)
run=pathlib.Path(__file__).resolve().parent.parent
(run/'prelaunch/relay-environment-launch.json').write_text(p.stdout)
print(p.stdout)
