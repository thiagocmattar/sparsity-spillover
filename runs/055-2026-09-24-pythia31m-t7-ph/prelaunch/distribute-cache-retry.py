"""Retry peer transport using container-local key permissions and a US-CA relay."""
import concurrent.futures,importlib.util,json,pathlib,shlex,subprocess,time
p=pathlib.Path('/workspace/run055-control/distribute-cache.py')
s=importlib.util.spec_from_file_location('distribution',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
old_ssh=m.ssh
def ssh(node):
    return ['/root/.ssh/run055-transfer-key' if x==str(m.CONTROL/'transfer-key') else x for x in old_ssh(node)]
m.ssh=ssh
def copy(i):
    tick=time.time();target='tar -xf - -C '+str(m.ROOT)
    cmd=ssh(m.NODES[i])+[target] if i==1 else ssh(m.NODES[1])+[shlex.join(ssh(m.NODES[i])+[target])]
    with (m.CONTROL/'input-data.tar').open('rb') as f:subprocess.run(cmd,stdin=f,check=True,timeout=900)
    m.emit(stage='cache_copied_retry',node=i,seconds=time.time()-tick)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    for f in [pool.submit(copy,i) for i in (1,2)]:f.result()
result=m.RUN/'prelaunch/lifecycle-cloud-001/result.json'
for attempt in range(240):
    if result.exists() and json.loads(result.read_text())['status']=='passed':break
    time.sleep(5)
else:raise RuntimeError('Lifecycle not passed within bounded wait')
for i in range(3):
    if i==0:subprocess.run(['bash','-lc',m.command(i)],check=True)
    else:subprocess.run(ssh(m.NODES[i])+[m.command(i)],capture_output=True,text=True,check=True,timeout=120)
    m.emit(stage='pipeline_started',node=i,conditions=m.NODES[i]['conditions'])
subprocess.run(ssh(m.NODES[1])+['rm /root/.ssh/run055-transfer-key'],check=True)
pathlib.Path('/root/.ssh/run055-transfer-key').unlink()
(m.CONTROL/'transfer-key').unlink()
(m.CONTROL/'distribution-ready').touch()
m.emit(stage='complete',private_transfer_keys_removed=True)
