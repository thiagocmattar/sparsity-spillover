"""Infrastructure retry: resume immutable bundle with four independent streams."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shlex
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from runio import module,read,write,record
cloud=module('run065_cloud',ROOT/'06_cloud.py')

def main():
    receipt=read(ROOT/'prelaunch/bundle-receipt.json');source=ROOT/'bundles/input-001.tar.gz';start=time.monotonic()
    client,lease=cloud.connect();control=cloud.CONTROL;remote=cloud.REMOTE
    try:
        offset=int(cloud.execute(client,'stat -c %s /opt/run065-input.tar.gz').strip())
        if not 0<offset<receipt['bytes']:raise ValueError('Invalid resume offset')
        prefix=hashlib.sha256()
        with source.open('rb') as f:
            remaining=offset
            while remaining:
                chunk=f.read(min(8*1024**2,remaining));prefix.update(chunk);remaining-=len(chunk)
        if cloud.execute(client,'sha256sum /opt/run065-input.tar.gz').split()[0]!=prefix.hexdigest():raise ValueError('Prefix mismatch')
        write(ROOT/'prelaunch/ingress-attempts/001/result.json',{'status':'superseded','bytes_verified':offset,'prefix_sha256':prefix.hexdigest(),'reason':'Per-connection transfer rate; four concurrent resumed streams; scientific bytes unchanged'})
        def transfer(i):
            a=offset+(receipt['bytes']-offset)*i//4;b=offset+(receipt['bytes']-offset)*(i+1)//4
            connection,_=cloud.connect();digest=hashlib.sha256();sent=0;next_report=20*1024**2
            try:
                path=f'/opt/run065-tail-{i}.part';stdin,stdout,stderr=connection.exec_command('cat > '+path,timeout=60)
                with source.open('rb') as f:
                    f.seek(a)
                    while sent<b-a:
                        chunk=f.read(min(1024**2,b-a-sent));digest.update(chunk);stdin.channel.sendall(chunk);sent+=len(chunk)
                        if sent>=next_report:
                            print(json.dumps({'stream':i,'bytes':sent,'total':b-a,'seconds':time.monotonic()-start}),flush=True);next_report+=20*1024**2
                stdin.channel.shutdown_write();stdout.read()
                if stdout.channel.recv_exit_status():raise RuntimeError(stderr.read().decode(errors='replace'))
                if cloud.execute(connection,'sha256sum '+path).split()[0]!=digest.hexdigest():raise ValueError('Part hash')
                return {'stream':i,'offset':a,'bytes':sent,'sha256':digest.hexdigest()}
            finally:connection.close()
        with ThreadPoolExecutor(max_workers=4) as pool:parts=list(pool.map(transfer,range(4)))
        cloud.execute(client,'cat /opt/run065-input.tar.gz /opt/run065-tail-0.part /opt/run065-tail-1.part /opt/run065-tail-2.part /opt/run065-tail-3.part > /opt/run065-input-full.tar.gz',timeout=180)
        if cloud.execute(client,'sha256sum /opt/run065-input-full.tar.gz').split()[0]!=receipt['sha256']:raise ValueError('Assembled hash')
        cloud.execute(client,f'mv /opt/run065-input-full.tar.gz /opt/run065-input.tar.gz && test ! -e {remote} && cp /opt/run065-input.tar.gz /workspace/run065-input.tar.gz && mkdir {remote} && tar -xf /opt/run065-input.tar.gz -C {remote}',timeout=300)
        if cloud.execute(client,'sha256sum /workspace/run065-input.tar.gz').split()[0]!=receipt['sha256']:raise ValueError('Persistent hash')
        check=f"""python3 - <<'PY'
from pathlib import Path
import hashlib,json
root=Path('{remote}')
for row in json.loads((root/'prelaunch/bundle-inventory.json').read_text())['files']:
 p=root/row['path']
 if p.stat().st_size!=row['bytes'] or hashlib.file_digest(p.open('rb'),'sha256').hexdigest()!=row['sha256']:raise ValueError(row['path'])
print('Every bundle member verified')
PY"""
        print(cloud.execute(client,check,timeout=300),flush=True)
        with client.open_sftp() as sftp:
            with sftp.open(remote+'/prelaunch/deployment.json','w') as f:f.write(json.dumps({'lease':lease,'bundle':receipt,'source_commit':lease['source_commit'],'ingress_attempt':2}))
        wrapper=f'export RUN065_DEADLINE={lease["deadline_epoch"]}; bash {remote}/04_setup.sh; code=$?; printf "%s\\n" "$code" > {control}/pipeline.exit'
        cloud.execute(client,f'nohup setsid bash -c {shlex.quote(wrapper)} > {control}/pipeline.log 2>&1 < /dev/null & echo $! > {control}/pipeline.pid')
        write(ROOT/'prelaunch/ingress-attempts/002/result.json',{'status':'verified','offset':offset,'parts':parts,'seconds':time.monotonic()-start,'archive_sha256':receipt['sha256'],'members_verified':receipt['files'],'script':record(Path(__file__)),'detached':True})
        print('Detached authoritative setup/profile pipeline started',flush=True)
    finally:client.close()

if __name__=='__main__':main()
