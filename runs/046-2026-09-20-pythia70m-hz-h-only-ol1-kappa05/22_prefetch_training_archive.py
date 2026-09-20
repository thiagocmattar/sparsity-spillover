"""Stream the append-only archive during compression, then verify with Run046's retriever."""
import importlib.util
from pathlib import Path
import json
import shlex
import socket
import subprocess
import sys
import time

RUN=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('run046_remote',RUN/'10_remote.py')
remote=importlib.util.module_from_spec(spec);spec.loader.exec_module(remote)


def main():
    staging=RUN/'retrieval';staging.mkdir(exist_ok=True)
    part=staging/'training-001.tar.gz.part'
    if (staging/'training-001.tar.gz').exists():
        raise FileExistsError('Use the final verifier for an already downloaded archive')
    offset=part.stat().st_size if part.exists() else 0
    root=remote.REMOTE+'/runs/'+RUN.name
    producer=f'''from pathlib import Path
import sys,time
root=Path({root!r})
assert Path('/workspace/run046-control/training-verified').exists()
path=root/'transfer/training-001.tar.gz'
assert path.stat().st_size>={offset}
with path.open('rb') as handle:
 handle.seek({offset})
 while True:
  chunk=handle.read(1024**2)
  if chunk:
   sys.stdout.buffer.write(chunk);sys.stdout.buffer.flush()
  elif (root/'transfer/training-receipt-001.json').exists():
   break
  else:time.sleep(1)
'''
    client,_=remote.connect('training')
    started=time.monotonic();last=started;done=offset
    try:
        transport=client.get_transport()
        transport.sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
        transport.sock.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,8*1024**2)
        channel=transport.open_session(window_size=64*1024**2,max_packet_size=1024**2)
        channel.settimeout(120)
        channel.exec_command('python3 -u -c '+shlex.quote(producer))
        with part.open('ab') as handle:
            while chunk:=channel.recv(1024**2):
                handle.write(chunk);done+=len(chunk)
                if time.monotonic()-last>=30:
                    print(json.dumps({'prefetched_bytes':done,'MiB_per_second':(done-offset)/1024**2/(time.monotonic()-started)}),flush=True)
                    last=time.monotonic()
        assert channel.recv_exit_status()==0,'Prefix retained for the resumable final retriever'
        channel.close()
    finally:client.close()
    print(json.dumps({'archive_stream_complete':True,'bytes':done,'seconds':time.monotonic()-started}),flush=True)
    subprocess.run([sys.executable,str(RUN/'16_retrieve_training.py')],check=True)


if __name__=='__main__':main()
