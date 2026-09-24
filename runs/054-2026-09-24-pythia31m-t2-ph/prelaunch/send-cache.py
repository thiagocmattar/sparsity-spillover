import pathlib,json,subprocess,time
run=pathlib.Path(__file__).resolve().parent.parent
n=json.loads((run/'prelaunch/parallel-connections-001.json').read_text())[1];s=n['ssh']
key=str(pathlib.Path.home()/'.runpod/ssh/runpodctl-ssh-key')
b=['ssh','-o','BatchMode=yes','-i',key,'-p',str(s['port']),s['username']+'@'+s['host']]
p=subprocess.Popen(b+['cat > /workspace/run054-control/input-data.tar'],stdin=subprocess.PIPE)
started=time.time();sent=0
with (run/'prelaunch/input-data.tar').open('rb') as f:
 while block:=f.read(8*1024*1024):
  p.stdin.write(block);sent+=len(block)
  if sent%(256*1024*1024)==0: print(json.dumps(dict(bytes=sent,seconds=time.time()-started,MBps=sent/1e6/(time.time()-started))),flush=True)
p.stdin.close();code=p.wait();print(json.dumps(dict(status='uploaded' if code==0 else 'failed',bytes=sent,seconds=time.time()-started)),flush=True)
