import concurrent.futures,json,pathlib,subprocess,shutil,time
run=pathlib.Path(__file__).resolve().parent.parent
archive=run/'prelaunch/input-source-3faa9d46.tar'
shutil.copyfile(run/'prelaunch/input-source.tar',archive)
nodes=json.loads((run/'prelaunch/parallel-connections-001.json').read_text())
key=str(pathlib.Path.home()/'.runpod/ssh/runpodctl-ssh-key')
def work(n):
 s=n['ssh'];base=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-i',key,'-p',str(s['port']),s['username']+'@'+s['host']]
 started=time.time()
 with archive.open('rb') as f:
  subprocess.run(base+['tar -xf - -C /workspace/sparsity-spillover'],stdin=f,check=True,timeout=600)
 command='nohup bash /workspace/sparsity-spillover/runs/054-2026-09-24-pythia31m-t2-ph/00_setup_remote.sh > /workspace/run054-control/setup.log 2>&1 < /dev/null & echo SETUP_STARTED'
 p=subprocess.run(base+[command],capture_output=True,text=True,check=True,timeout=30)
 return dict(pod=n['name'],seconds=time.time()-started,result=p.stdout)
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
 for f in concurrent.futures.as_completed([pool.submit(work,n) for n in nodes]):
  try: print(json.dumps(f.result()),flush=True)
  except Exception as e: print(json.dumps(dict(error=str(e))),flush=True)
