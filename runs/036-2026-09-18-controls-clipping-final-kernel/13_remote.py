"""Scoped read-only monitoring or explicit detached launch for this one Pod."""
import argparse
import json
from remote_io import HERE, connect, execute


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['status','launch','retrieve'])
    args=p.parse_args();c=connect()
    if args.action=='status':
        command="""cd /workspace/run036
for f in runtime/setup.exit runtime/extract.exit runtime/execute.exit; do if [ -f "$f" ]; then echo "$f: $(cat "$f")"; fi; done
tail -n 4 runtime/setup.log
if [ -f runtime/execute.log ]; then tail -n 3 runtime/execute.log; fi
python3 - <<'PY'
import json,time
from pathlib import Path
paths=list(Path('artifacts/attempts').glob('*/status.json'))
if paths:
 p=max(paths,key=lambda p:p.stat().st_mtime);d=json.loads(p.read_text());print(json.dumps({'latest':str(p),'stale_seconds':time.time()-p.stat().st_mtime,'status':d}))
for phase in ['preflight','scientific']:
 p=Path('artifacts')/phase/'summary-001.json'
 if p.exists():
  rows=json.loads(p.read_text());print(json.dumps({'phase':phase,'done':len(rows),'qualified':sum(bool(r['qualified']) for r in rows),'seconds':sum(r['seconds'] for r in rows)}))
PY
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
df -h /workspace | tail -1
"""
        print(execute(c,command),flush=True)
    elif args.action=='launch':
        lease=json.loads((HERE/'prelaunch/lease-001.json').read_text())
        deadline=lease['deadline_epoch']
        command=f"cd /workspace/run036; test -f runtime/setup-complete; test ! -e runtime/execute.log; nohup bash -c 'bash 05_execute.sh {deadline}; echo $? > runtime/execute.exit' > runtime/execute.log 2>&1 < /dev/null & echo $!"
        print(execute(c,command),flush=True)
    else:
        s=c.open_sftp();out=HERE/'retrieval';out.mkdir(exist_ok=True)
        for name in ['receipt-001.json','output-001.tar.gz']:
            s.get('/workspace/run036/transfer/'+name,str(out/name))
        s.close();print('Retrieval complete; run 09_verify_retrieval.py before teardown',flush=True)
    c.close()


if __name__=='__main__':main()
