"""Run052 native OpenSSH transport; exact Pod identity from the approved lease."""
import argparse,json,shlex,subprocess,time
from pathlib import Path

HERE=Path(__file__).resolve().parent
RUN=HERE.parent
ROOT=RUN.parents[1]


def info():
    lease=json.loads((HERE/'lease-001.json').read_text(encoding='utf-8-sig'))
    value=json.loads((HERE/'ssh-001.json').read_text(encoding='utf-8-sig'))
    assert value['id']==lease['pod']['id'] and value['name']==lease['pod']['name']
    return value


def ssh():
    value=info()
    return ['ssh','-i',value['ssh_key']['path'],'-p',str(value['port']),'-o','BatchMode=yes',
            '-o','StrictHostKeyChecking=accept-new','-o','UserKnownHostsFile='+str(HERE/'known_hosts'),
            '-o','ConnectTimeout=20','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=4','root@'+value['ip']]


def execute(command,timeout=60):
    result=subprocess.run(ssh()+[command],capture_output=True,text=True,encoding='utf-8',timeout=timeout)
    if result.returncode:raise RuntimeError(f'SSH exit{result.returncode}: {result.stdout}\n{result.stderr}')
    return result.stdout


def upload(local,remote):
    local=Path(local);start=time.monotonic();last=start;done=0
    with local.open('rb') as source:
        process=subprocess.Popen(ssh()+['cat > '+shlex.quote(remote)],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        try:
            for chunk in iter(lambda:source.read(1024*1024),b''):
                process.stdin.write(chunk);done+=len(chunk);now=time.monotonic()
                if now-last>=15:
                    print(json.dumps({'bytes':done,'total':local.stat().st_size,'MB_per_second':done/1e6/(now-start),'remaining_seconds':(local.stat().st_size-done)/(done/(now-start))}),flush=True);last=now
            process.stdin.close();error=process.stderr.read().decode();code=process.wait()
            if code:raise RuntimeError(f'Upload exit{code}: {error}')
        except BaseException:process.kill();process.wait();raise
    print(json.dumps({'uploaded':local.name,'bytes':done,'seconds':time.monotonic()-start}),flush=True)


def download(remote,local):
    local=Path(local);local.parent.mkdir(parents=True,exist_ok=True)
    if local.exists():raise FileExistsError(local)
    with local.open('wb') as target:
        result=subprocess.run(ssh()+['cat '+shlex.quote(remote)],stdout=target,stderr=subprocess.PIPE)
    if result.returncode:raise RuntimeError(result.stderr.decode())
    print(json.dumps({'downloaded':local.name,'bytes':local.stat().st_size}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('exec','upload','download'));p.add_argument('first');p.add_argument('second',nargs='?');a=p.parse_args()
    if a.action=='exec':print(execute(a.first),end='')
    elif a.action=='upload':upload(a.first,a.second)
    else:download(a.first,a.second)
