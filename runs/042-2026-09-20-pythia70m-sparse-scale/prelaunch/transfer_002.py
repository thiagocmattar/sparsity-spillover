"""Resumeable bounded independent SSH streams for the owned Pod's input archive."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import threading
import time
import remote

RUN = Path(__file__).resolve().parent.parent
PART = 8*1024**2
auth = threading.Semaphore(4)
lock = threading.Lock()
progress = {'bytes':0,'parts':0}
started = time.monotonic()
receipt = json.loads((RUN/'bundles/input-001-receipt.json').read_text())

def worker(indices):
    with auth: client = remote.connect()
    sftp = client.open_sftp()
    with (RUN/receipt['path']).open('rb') as source:
        for i in indices:
            source.seek(i*PART)
            data = source.read(PART)
            destination = f'/workspace/run042/runtime/input-parts-002/{i:04d}'
            try: offset = sftp.stat(destination).st_size
            except FileNotFoundError: offset = 0
            if not 0 <= offset <= len(data): raise ValueError('Invalid partial size')
            if offset < len(data):
                with sftp.file(destination,'ab' if offset else 'wb') as output:
                    output.set_pipelined(True)
                    for pos in range(offset,len(data),256*1024): output.write(data[pos:pos+256*1024])
            with lock:
                progress['bytes'] += len(data)
                progress['parts'] += 1
    client.close()

def main():
    client = remote.connect()
    remote.execute(client,'mkdir -p /workspace/run042/runtime/input-parts-002')
    count = (receipt['bytes']+PART-1)//PART
    with concurrent.futures.ThreadPoolExecutor(32) as executor:
        pending = {executor.submit(worker,range(i,count,32)) for i in range(32)}
        while pending:
            done,pending = concurrent.futures.wait(pending,timeout=30,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done: future.result()
            print(json.dumps({**progress,'seconds':time.monotonic()-started,'total':receipt['bytes']}),flush=True)
    command = "cat /workspace/run042/runtime/input-parts-002/[0-9][0-9][0-9][0-9] > /workspace/run042-input-002.tar.gz && "
    command += "echo '"+receipt['sha256']+"  /workspace/run042-input-002.tar.gz' | sha256sum -c - && "
    command += 'tar -xzf /workspace/run042-input-002.tar.gz -C /workspace/run042'
    print(remote.execute(client,command,timeout=180),flush=True)
    print(remote.execute(client,'cd /workspace/run042 && python3 01_prepare.py verify',timeout=90),flush=True)
    (RUN/'prelaunch/upload-002.json').write_text(json.dumps({'archive':receipt,'seconds':time.monotonic()-started,'verified':True,'streams':32},indent=2)+'\n')
    client.close()

if __name__ == '__main__': main()
