"""Resume the one scoped input archive by SHA256-verified 8MiB SFTP blocks.

Both competing transfer processes must be stopped first. This changes transport
only, never the model or data. No private key or relay code enters the receipt.
"""
import argparse
import hashlib
import json
import shlex
import time
import paramiko
from common import RUN, R27, record, write_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--host',required=True)
    p.add_argument('--port',type=int,required=True)
    p.add_argument('--key',required=True)
    p.add_argument('--known-hosts',required=True)
    p.add_argument('--receipt',required=True)
    a=p.parse_args()
    path=R27/'bundles/inputs-001/payload.tar.gz'
    target='/workspace/payload.tar.gz'
    chunk=8*1024*1024
    started=time.monotonic()
    identity=record(path)
    if identity['sha256']!='044dd4115f1bd4066dba11ae4e1b8305e3e939f6376b38562186b1df95ef9f19':raise ValueError('Wrong original input archive')
    client=paramiko.SSHClient();client.load_host_keys(a.known_hosts)
    client.connect(a.host,port=a.port,username='root',key_filename=a.key,look_for_keys=False,allow_agent=False,timeout=30)
    client.get_transport().set_keepalive(30)
    def remote(command):
        _,out,err=client.exec_command(command,timeout=120)
        result=out.read().decode();error=err.read().decode()
        if out.channel.recv_exit_status()!=0:raise RuntimeError(error)
        return result
    probe=f'import hashlib,json,os; p={target!r}; f=open(p,"rb"); print(json.dumps(dict(size=os.stat(p).st_size,hashes=[hashlib.sha256(b).hexdigest() for b in iter(lambda:f.read({chunk}),b"")]))); f.close()'
    state=json.loads(remote('python3 -c '+shlex.quote(probe)))
    if state['size']!=identity['bytes']:raise ValueError('Expected preallocated same-size archive')
    missing=[]
    with path.open('rb') as src:
        for i,digest in enumerate(state['hashes']):
            data=src.read(chunk)
            if hashlib.sha256(data).hexdigest()!=digest:missing.append((i,len(data)))
    needed=sum(size for _,size in missing)
    receipt={'status':'transferring','source':identity,'remote':target,'block_bytes':chunk,'blocks_to_replace':missing,'bytes_to_transfer':needed}
    write_json(RUN/'launch-control'/a.receipt,receipt)
    print(json.dumps({'stage':'resume','total_blocks':len(state['hashes']),'mismatched_blocks':len(missing),'bytes_to_transfer':needed}),flush=True)
    sftp=paramiko.SFTPClient.from_transport(client.get_transport(),window_size=64*1024*1024,max_packet_size=256*1024)
    transferred=0;before=time.monotonic()
    with path.open('rb') as src,sftp.open(target,'r+b',bufsize=1024*1024) as dst:
        dst.set_pipelined(True)
        for i,size in missing:
            src.seek(i*chunk);data=src.read(size)
            dst.seek(i*chunk);dst.write(data);dst.flush()
            transferred+=size
            elapsed=time.monotonic()-before
            print(json.dumps({'stage':'transfer','bytes':transferred,'total':needed,'MB_per_second':transferred/max(elapsed,.001)/1e6,'ETC_seconds':(needed-transferred)*elapsed/max(transferred,1)}),flush=True)
    actual=remote('sha256sum '+target).split()[0]
    if actual!=identity['sha256']:raise ValueError('Resumed archive hash mismatch')
    receipt.update(status='verified',remote_sha256=actual,elapsed_seconds=time.monotonic()-started)
    write_json(RUN/'launch-control'/a.receipt,receipt)
    sftp.close();client.close();print(json.dumps({'stage':'verified','sha256':actual}),flush=True)


if __name__=='__main__':main()
