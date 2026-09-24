"""Use the established US seed to bypass the stalled workstation-to-Iceland link."""
import json,pathlib,subprocess,tarfile
run=pathlib.Path(__file__).resolve().parent.parent
with tarfile.open(run/'prelaunch/input-training.tar') as t:names=t.getnames()
code='NAMES='+repr(names)+'\n'+'''
import pathlib,tarfile,subprocess,json,time
root=pathlib.Path('/workspace/sparsity-spillover');c=pathlib.Path('/workspace/run055-control')
ssh=['ssh','-i','/root/.ssh/run055-transfer-key','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','StrictHostKeyChecking=accept-new','-p','13409','root@213.181.104.53']
archive=c/'replacement-e-source.tar'
with tarfile.open(archive,'w') as f:
 for name in NAMES:f.add(root/name,arcname=name,recursive=False)
with archive.open('rb') as f:subprocess.run(ssh+['tar -xf - -C /workspace/sparsity-spillover'],stdin=f,check=True,timeout=600)
target_code="""import subprocess,pathlib,json
c=pathlib.Path('/workspace/run055-control')
with (c/'setup.log').open('x') as f:
 p=subprocess.Popen(['bash','/workspace/sparsity-spillover/runs/055-2026-09-24-pythia31m-t7-ph/00_setup_remote.sh'],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(setup_pid=p.pid)))
"""
subprocess.run(ssh+['python3 -'],input=target_code,text=True,check=True,timeout=30)
with (c/'input-data.tar').open('rb') as f:
 compressor=subprocess.Popen(['gzip','-1','-c'],stdin=f,stdout=subprocess.PIPE)
 try:
  subprocess.run(ssh+['gzip -dc > /workspace/run055-control/input-data.tar'],stdin=compressor.stdout,check=True,timeout=900)
  compressor.stdout.close();assert compressor.wait(timeout=30)==0
 finally:
  if compressor.poll() is None:compressor.kill();compressor.wait()
target_code="""import pathlib,hashlib,tarfile,time,subprocess,os,json
c=pathlib.Path('/workspace/run055-control');root=pathlib.Path('/workspace/sparsity-spillover')
a=c/'input-data.tar'
with a.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()=='86415e9b8b3f32f4a0fbfb210d37fb4bd1fea66442a15b311c1f44ff2b2652b5'
with tarfile.open(a) as f:f.extractall(root,filter='data')
while not (c/'environment-ready').exists():
 assert time.time()<1790286236-7200
 time.sleep(20)
r=root/'runs/055-2026-09-24-pythia31m-t7-ph';python='/workspace/run055-venv/bin/python'
subprocess.run([python,str(r/'12_check_inputs.py')],check=True)
args=[python,str(r/'15_parallel_train.py'),'--workers','a7-h-ol1-kappa-0p1','--gpus','0','--deadline','2026-09-24T21:43:56Z','--tag','parallel-4-002']
with (c/'replacement-pipeline.log').open('x') as f:
 p=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'))
(c/'replacement-launch.json').write_text(json.dumps(dict(pid=p.pid,command=args,unix_time=time.time()),indent=2))
print(json.dumps(dict(pid=p.pid,status='launched')))
"""
subprocess.run(ssh+['python3 -'],input=target_code,text=True,check=True,timeout=600)
'''
deploy='SCRIPT='+repr(code)+'\n'+'''
import pathlib,subprocess,json
c=pathlib.Path('/workspace/run055-control');p=c/'relay-replacement.py';p.write_text(SCRIPT)
with (c/'relay-replacement.log').open('x') as f:
 child=subprocess.Popen(['python3',str(p)],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=child.pid,status='relay_started')))
'''
ssh=['ssh','-i',str(pathlib.Path.home()/'.runpod/ssh/runpodctl-ssh-key'),'-o','BatchMode=yes','-p','27548','root@103.196.86.181']
p=subprocess.run(ssh+['python3 -'],input=deploy,text=True,capture_output=True,check=True)
(run/'prelaunch/relay-replacement-launch.json').write_text(p.stdout)
print(p.stdout)
