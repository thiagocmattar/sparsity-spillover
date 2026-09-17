"""Resume the verified archive prefix using current MCP-observed SSH addresses."""
import concurrent.futures
import importlib.util
import json
from pathlib import Path
import time
import paramiko

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('retrieve032',HERE/'12_retrieve_worker.py')
retrieve032=importlib.util.module_from_spec(spec)
spec.loader.exec_module(retrieve032)


def recover(pod):
    path=HERE/'retrieval'/pod['id']/'results.tar'
    path.parent.mkdir(parents=True,exist_ok=True)
    offset=path.stat().st_size if path.exists() else 0
    started=time.monotonic()
    with retrieve032.remote.connect(pod['id']) as client:
        sftp=paramiko.SFTPClient.from_transport(client.get_transport(),window_size=32*1024**2,max_packet_size=32768)
        sftp.get_channel().settimeout(30)
        remote='/workspace/run032-results.tar'
        total=sftp.stat(remote).st_size
        print(json.dumps({'pod':pod['id'],'resume_bytes':offset,'total':total}),flush=True)
        with sftp.open(remote,'rb') as source,path.open('ab') as target:
            source.seek(offset)
            source.prefetch(file_size=total,max_concurrent_requests=32)
            copied=offset
            last=copied
            while copied<total:
                data=source.read(min(1024**2,total-copied))
                if not data:raise RuntimeError('Premature EOF')
                target.write(data)
                copied+=len(data)
                if copied-last>=100*1024**2:
                    print(json.dumps({'pod':pod['id'],'downloaded_bytes':copied,'MB_per_s':(copied-offset)/1e6/(time.monotonic()-started)}),flush=True)
                    last=copied
        sftp.close()
    retrieve032.retrieve(pod,archive_path=path)


if __name__=='__main__':
    pods=json.loads((HERE/'prelaunch/pods.json').read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(recover,p) for p in pods if p['id']!='8o1uyxgh6vb6ym']
        for future in concurrent.futures.as_completed(futures):
            try:future.result()
            except Exception as error:print(json.dumps({'error':str(error)}),flush=True)
