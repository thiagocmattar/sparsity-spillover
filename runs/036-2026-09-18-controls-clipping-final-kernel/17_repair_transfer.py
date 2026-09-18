"""Repair mismatched 8MiB ranges of this one frozen bundle after slow relay."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import time
import threading
import paramiko
from remote_io import HERE, connect, execute

REMOTE='/workspace/run036/relay-bundle-001/input-001.tar'
CHUNK=8*1024*1024
CONNECT_LOCK=threading.Lock()


def main():
    source=HERE/'bundles/input-001.tar'
    receipt=json.loads((HERE/'bundles/input-001.receipt.json').read_text())
    local=[]
    with source.open('rb') as f:
        while block:=f.read(CHUNK):local.append(hashlib.sha256(block).hexdigest())
    c=connect()
    # The recorded receiver PID belongs to this conversation's transfer only.
    script=f"""python3 - <<'PY'
import hashlib,json
from pathlib import Path
p=Path('{REMOTE}'); hashes=[]
with p.open('rb') as f:
 while block:=f.read({CHUNK}):hashes.append(hashlib.sha256(block).hexdigest())
print(json.dumps(hashes))
PY"""
    remote=json.loads(execute(c,script,timeout=60));c.close()
    assert len(local)==len(remote)
    missing=[i for i,(a,b) in enumerate(zip(local,remote)) if a!=b]
    print(json.dumps({'mismatched_chunks':len(missing),'total_chunks':len(local)}),flush=True)
    started=time.monotonic()
    # Disjoint ranges; each connection exclusively owns the chunks in its bucket.
    def worker(bucket):
        # The inherited transport refreshes one shared SSH-info cache file.
        with CONNECT_LOCK:
            client=connect()
        s=paramiko.SFTPClient.from_transport(client.get_transport(),window_size=8*1024*1024,max_packet_size=32768)
        s.get_channel().settimeout(60)
        done=0
        try:
            with source.open('rb') as f,s.open(REMOTE,'r+b') as out:
                out.set_pipelined(True)
                for index in bucket:
                    f.seek(index*CHUNK);data=f.read(CHUNK);out.seek(index*CHUNK);out.write(data);done+=len(data)
        finally:s.close();client.close()
        print(json.dumps({'worker_bytes':done,'seconds':time.monotonic()-started}),flush=True)
        return done
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(worker,missing[i::4]) for i in range(4) if missing[i::4]]
        repaired=sum(f.result() for f in as_completed(futures))
    c=connect()
    checked=execute(c,f"sha256sum {REMOTE}",timeout=60).split()[0]
    assert checked==receipt['sha256']
    report={'status':'complete_verified','mismatched_chunks':missing,'repaired_bytes':repaired,
        'elapsed_seconds':time.monotonic()-started,'archive_sha256':checked,
        'method':'Four disjoint SFTP writers with 60s channel timeouts; all other ranges checked by SHA256'}
    (HERE/'prelaunch/transfer-repair.json').write_text(json.dumps(report,indent=2)+'\n')
    s=c.open_sftp()
    with s.open('/workspace/run036/runtime/transfer-repair.json','w') as f:f.write(json.dumps(report,indent=2)+'\n')
    with s.open('/workspace/run036/runtime/relay-bundle-001.exit','w') as f:f.write('0\n')
    s.close();c.close();print(json.dumps(report),flush=True)


if __name__=='__main__':main()
