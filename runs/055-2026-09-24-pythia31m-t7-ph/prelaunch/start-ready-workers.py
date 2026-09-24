"""Start each verified host independently while existing peer copies finish."""
import concurrent.futures,importlib.util,json,pathlib,shlex,subprocess,time
p=pathlib.Path('/workspace/run055-control/distribute-cache.py')
s=importlib.util.spec_from_file_location('distribution',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def ssh(node):
    return ['/root/.ssh/run055-transfer-key' if x==str(m.CONTROL/'transfer-key') else x for x in m.ssh(node)]
def remote(i,command):
    if i==0:return ['bash','-lc',command]
    if i==1:return ssh(m.NODES[i])+[command]
    return ssh(m.NODES[1])+[shlex.join(ssh(m.NODES[i])+[command])]
def start(i):
    check=('test -f /workspace/run055-control/environment-ready && '+
        'test "$(stat -c %s '+str(m.ROOT)+'/data/tokenized/minipile-pythia-14m-full/train/tokens.int32.bin 2>/dev/null)" = 5966845664 && '+
        'test "$(stat -c %s '+str(m.ROOT)+'/data/tokenized/minipile-pythia-14m-full/validation/tokens.int32.bin 2>/dev/null)" = 2774672')
    while time.time()<m.LAST_START:
        probe=subprocess.run(remote(i,check),capture_output=True,text=True,timeout=45)
        if probe.returncode==0:break
        time.sleep(30)
    else:raise TimeoutError('Cache not ready before retrieval reserve')
    # This command recomputes every input hash before the detached pipeline starts.
    assert not (m.RUN/f'artifacts/pipeline/parallel-{i}-001/status.json').exists() if i==0 else True
    result=subprocess.run(remote(i,m.command(i)),capture_output=True,text=True,check=True,timeout=180)
    m.emit(stage='pipeline_started',node=i,conditions=m.NODES[i]['conditions'],output=result.stdout)
assert json.loads((m.RUN/'prelaunch/lifecycle-cloud-001/result.json').read_text())['status']=='passed'
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    for future in [pool.submit(start,i) for i in range(3)]:future.result()
subprocess.run(ssh(m.NODES[1])+['rm /root/.ssh/run055-transfer-key'],check=True)
pathlib.Path('/root/.ssh/run055-transfer-key').unlink()
(m.CONTROL/'transfer-key').unlink()
(m.CONTROL/'distribution-ready').touch()
m.emit(stage='complete',private_transfer_keys_removed=True)
