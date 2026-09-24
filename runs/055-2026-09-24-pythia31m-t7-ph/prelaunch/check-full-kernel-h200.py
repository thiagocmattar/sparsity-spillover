"""Extra full-validation forward check on an idle H200, excluded from final timing."""
import pathlib,subprocess
worker='''import pathlib,json,subprocess,os
r=pathlib.Path('/workspace/sparsity-spillover/runs/055-2026-09-24-pythia31m-t7-ph');c=pathlib.Path('/workspace/run055-control')
rows=[v for p in (r/'artifacts').glob('verification-*.json') for v in json.loads(p.read_text()).get('conditions',[]) if v['condition']['id']=='a7-h-ol1-kappa-0' and v['status']=='verified']
row,=rows
checkpoint=r/'artifacts/attempts'/row['attempt']/row['final_checkpoint']['path']
env=dict(os.environ,PATH='/workspace/run055-venv/bin:'+os.environ['PATH'],CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
args=['/workspace/run055-venv/bin/python',str(r/'latency/01_benchmark.py'),'--checkpoint',str(checkpoint),'--condition','calibration-a7-h-ol1-kappa-0','--attempt','h200-full-k0-001','--replicate','2','--local-calibration']
result=dict(scientific_measurement=False,scope='all338 blocks; H200 forward preflight only',command=args)
try:
 p=subprocess.run(args,env=env,timeout=1200);result['returncode']=p.returncode
 result['manifest']=json.loads((r/'latency/artifacts/attempts/h200-full-k0-001/manifest.json').read_text())
finally:
 (c/'h200-full-exit.json').write_text(json.dumps(result,indent=2))
'''
compile(worker,'h200-full-worker','exec')
code='WORKER='+repr(worker)+'\n'+'''
import pathlib,json,subprocess
c=pathlib.Path('/workspace/run055-control');script=c/'h200-full.py';script.write_text(WORKER)
assert json.loads((c/'h200-components-exit-003.json').read_text())['returncode']==0
with (c/'h200-full.log').open('x') as f:
 p=subprocess.Popen(['python3',str(script)],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=p.pid,scientific_measurement=False)))
'''
ssh=['ssh','-i',str(pathlib.Path.home()/'.runpod/ssh/runpodctl-ssh-key'),'-o','BatchMode=yes','-p','27548','root@103.196.86.181']
p=subprocess.run(ssh+['python3 -'],input=code,text=True,capture_output=True,check=True)
run=pathlib.Path(__file__).resolve().parent.parent
(run/'prelaunch/h200-full-launch.json').write_text(p.stdout)
print(p.stdout)
