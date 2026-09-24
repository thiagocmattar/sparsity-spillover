"""Stream the previously verified immutable cache archive to the seed Pod."""
import json,subprocess,time
from pathlib import Path
import importlib.util
p=Path(__file__).with_name("stage-cloud.py")
s=importlib.util.spec_from_file_location("stage_cloud",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
run,nodes,ssh=m.run,m.nodes,m.ssh
archive=run.parent/'054-2026-09-24-pythia31m-t2-ph/prelaunch/input-data.tar'
started=time.time();sent=0
p=subprocess.Popen(ssh(nodes[0])+['cat > /workspace/run055-control/input-data.tar'],stdin=subprocess.PIPE)
with archive.open('rb') as f:
    while block:=f.read(8*1024*1024):
        p.stdin.write(block);sent+=len(block)
        if sent%(256*1024*1024)==0:
            print(json.dumps(dict(bytes=sent,seconds=time.time()-started,MBps=sent/1e6/(time.time()-started))),flush=True)
p.stdin.close();code=p.wait()
assert code==0
print(json.dumps(dict(bytes=sent,seconds=time.time()-started,status='uploaded')),flush=True)
