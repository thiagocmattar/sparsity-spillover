"""Stage immutable packages and start detached environment setup on chosen Pods."""
import argparse, concurrent.futures, json, pathlib, subprocess, time
run=pathlib.Path(__file__).resolve().parent.parent
nodes=json.loads((run/'prelaunch/parallel-connections-001.json').read_text())
key=str(pathlib.Path.home()/'.runpod/ssh/runpodctl-ssh-key')

def ssh(n):
    s=n['ssh']
    return ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','StrictHostKeyChecking=accept-new','-i',key,'-p',str(s['port']),s['username']+'@'+s['host']]

def work(i):
    n=nodes[i]; started=time.time()
    subprocess.run(ssh(n)+['mkdir -p /workspace/sparsity-spillover /workspace/run055-control'],check=True,timeout=30)
    archive=run/'prelaunch'/('input-training.tar' if n['conditions'] else 'input-source.tar')
    with archive.open('rb') as f:
        child=subprocess.Popen(ssh(n)+['tar -xf - -C /workspace/sparsity-spillover'],stdin=subprocess.PIPE)
        try:
            while block:=f.read(8*1024*1024):child.stdin.write(block)
            child.stdin.close()
            if child.wait(timeout=600):raise RuntimeError('Source upload failed')
        finally:
            if child.poll() is None:child.kill();child.wait()
    command='nohup setsid bash /workspace/sparsity-spillover/runs/055-2026-09-24-pythia31m-t7-ph/00_setup_remote.sh > /workspace/run055-control/setup.log 2>&1 < /dev/null & echo SETUP_STARTED'
    p=subprocess.run(ssh(n)+[command],capture_output=True,text=True,check=True,timeout=30)
    return dict(pod=n['name'],seconds=time.time()-started,result=p.stdout)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('indices',nargs='+',type=int);a=p.parse_args()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(a.indices)) as pool:
        for f in concurrent.futures.as_completed([pool.submit(work,i) for i in a.indices]):
            print(json.dumps(f.result()),flush=True)
