"""Fine-block repair of this run's incomplete archive over one SSH byte stream."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import time

TARGET='/workspace/payload.tar.gz'
SIZE=1825710912
EXPECTED='044dd4115f1bd4066dba11ae4e1b8305e3e939f6376b38562186b1df95ef9f19'
BLOCK=32768


def receiver():
    if Path(TARGET).stat().st_size!=SIZE:raise ValueError('Unexpected archive size')
    with open(TARGET,'rb') as f:
        hashes=[hashlib.sha256(b).hexdigest() for b in iter(lambda:f.read(BLOCK),b'')]
    print(json.dumps(hashes),flush=True)
    source=sys.stdin.buffer
    def read(n):
        parts=[]
        while n:
            b=source.read(n)
            if not b:raise EOFError('Repair stream interrupted')
            parts.append(b);n-=len(b)
        return b''.join(parts)
    with open(TARGET,'r+b') as f:
        while True:
            offset,n=struct.unpack('!QI',read(12))
            if n==0:break
            if offset+n>SIZE or n>8*1024*1024:raise ValueError('Out-of-bounds repair')
            data=read(n);f.seek(offset);f.write(data)
    digest=hashlib.sha256()
    with open(TARGET,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):digest.update(b)
    if digest.hexdigest()!=EXPECTED:raise ValueError('Final archive hash mismatch')
    print(json.dumps({'status':'verified','sha256':digest.hexdigest()}),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--receiver',action='store_true')
    p.add_argument('--host');p.add_argument('--port',type=int);p.add_argument('--key');p.add_argument('--known-hosts');p.add_argument('--receipt')
    a=p.parse_args()
    if a.receiver:receiver();return
    import paramiko
    from common import RUN,R27,write_json,sha256
    source=R27/'bundles/inputs-001/payload.tar.gz'
    if sha256(source)!=EXPECTED:raise ValueError('Wrong source')
    client=paramiko.SSHClient();client.load_host_keys(a.known_hosts)
    client.connect(a.host,port=a.port,username='root',key_filename=a.key,look_for_keys=False,allow_agent=False,timeout=30)
    client.get_transport().set_keepalive(30)
    inp,out,err=client.exec_command('python3 /workspace/run028-stream-resume.py --receiver',timeout=600)
    hashes=json.loads(out.readline())
    missing=[]
    with source.open('rb') as f:
        for i,digest in enumerate(hashes):
            data=f.read(BLOCK)
            if hashlib.sha256(data).hexdigest()!=digest:
                if missing and missing[-1][0]+missing[-1][1]==i*BLOCK and missing[-1][1]+len(data)<=8*1024*1024:
                    missing[-1][1]+=len(data)
                else:missing.append([i*BLOCK,len(data)])
    needed=sum(n for _,n in missing)
    result={'status':'transferring','block_bytes':BLOCK,'segments':missing,'bytes_to_transfer':needed,'expected_sha256':EXPECTED}
    from common import record
    result['script']=record(__file__)
    write_json(RUN/'launch-control'/a.receipt,result)
    print(json.dumps({'stage':'repair','segments':len(missing),'bytes_to_transfer':needed}),flush=True)
    started=time.monotonic();sent=0;reported=0
    with source.open('rb') as f:
        for offset,n in missing:
            f.seek(offset);data=f.read(n)
            inp.channel.sendall(struct.pack('!QI',offset,n)+data)
            sent+=n
            if sent-reported>=8*1024*1024:
                elapsed=time.monotonic()-started
                print(json.dumps({'stage':'transfer','bytes':sent,'total':needed,'MB_per_second':sent/max(elapsed,.001)/1e6,'ETC_seconds':(needed-sent)*elapsed/max(sent,1)}),flush=True)
                reported=sent
    inp.channel.sendall(struct.pack('!QI',0,0));inp.channel.shutdown_write()
    terminal=json.loads(out.readline());errors=err.read().decode()
    if out.channel.recv_exit_status()!=0 or terminal.get('sha256')!=EXPECTED:raise RuntimeError(errors or 'Verification failed')
    result.update(terminal,elapsed_seconds=time.monotonic()-started)
    write_json(RUN/'launch-control'/a.receipt,result)
    client.close();print(json.dumps(terminal),flush=True)


if __name__=='__main__':main()
