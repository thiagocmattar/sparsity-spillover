"""Parallel SSH recovery; verify archive before the separate member-hash check."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import threading
import time
import remote

RUN=Path(__file__).resolve().parent.parent
PART=8*1024**2
AUTH=threading.Semaphore(4)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tag',default='001')
    args=parser.parse_args()
    assert args.tag.isalnum()
    dest=RUN/'retrieval';dest.mkdir(exist_ok=True)
    client=remote.connect();sftp=client.open_sftp()
    receipt_bytes=sftp.file(f'/workspace/run042/transfer/receipt-{args.tag}.json','rb').read()
    receipt=json.loads(receipt_bytes)
    assert receipt['path']==f'transfer/output-{args.tag}.tar.gz'
    receipt_path=dest/f'receipt-{args.tag}.json'
    if receipt_path.exists():assert receipt_path.read_bytes()==receipt_bytes
    else:receipt_path.write_bytes(receipt_bytes)
    client.close()
    parts=dest/f'parts-{args.tag}';parts.mkdir(exist_ok=True)
    count=(receipt['bytes']+PART-1)//PART
    started=time.monotonic();progress={'bytes':0,'parts':0};lock=threading.Lock()

    def worker(indices):
        with AUTH:connection=remote.connect()
        ftp=connection.open_sftp()
        with ftp.file('/workspace/run042/'+receipt['path'],'rb') as source:
            for index in indices:
                size=min(PART,receipt['bytes']-index*PART)
                path=parts/f'{index:05d}'
                if not path.exists() or path.stat().st_size!=size:
                    source.seek(index*PART)
                    # Independent streams avoid the single-connection transfer bottleneck.
                    with path.open('wb') as output:
                        left=size
                        while left:
                            chunk=source.read(min(left,256*1024))
                            if not chunk:raise EOFError('Incomplete remote archive')
                            output.write(chunk);left-=len(chunk)
                with lock:progress['bytes']+=size;progress['parts']+=1
        connection.close()

    with concurrent.futures.ThreadPoolExecutor(32) as executor:
        pending={executor.submit(worker,range(i,count,32)) for i in range(min(32,count))}
        while pending:
            done,pending=concurrent.futures.wait(pending,timeout=30,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:future.result()
            print(json.dumps({**progress,'total':receipt['bytes'],'seconds':time.monotonic()-started}),flush=True)
    archive=dest/f'output-{args.tag}.tar.gz';digest=hashlib.sha256()
    with archive.open('wb') as output:
        for index in range(count):
            data=(parts/f'{index:05d}').read_bytes();output.write(data);digest.update(data)
    assert archive.stat().st_size==receipt['bytes'] and digest.hexdigest()==receipt['sha256']
    result={'archive':receipt,'archive_hash_verified':True,'seconds':time.monotonic()-started,
            'streams':32,'member_verification_required':'09_verify_retrieval.py'}
    (RUN/f'prelaunch/download-{args.tag}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result,flush=True)


if __name__=='__main__':main()
