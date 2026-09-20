"""Three declared candidates, sequentially; no final-validation selection."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import os
import subprocess
import time
from io_utils import RUN, read, record, write


def main(deadline):
    diagnostic=read(RUN/'artifacts/scientific/summary-001.json')
    assert len(diagnostic)==54 and all(x['qualified'] for x in diagnostic)
    folder=RUN/'artifacts/development'
    folder.mkdir(parents=True,exist_ok=True)
    summary=folder/'summary-001.json'
    assert not summary.exists(), 'Keep previous development records'
    lock=RUN/'artifacts/controller.lock'
    with lock.open('x') as f: f.write(str(os.getpid()))
    rows=[]
    try:
        for candidate in ('opt001','opt002','opt003'):
            if time.time()+900+1200>deadline: break
            if candidate!='opt001':
                log=folder/('operator-'+candidate+'.log')
                with log.open('x') as f:
                    checked=subprocess.run([sys.executable,'-u',str(RUN/'candidates/check_joint.py'),candidate],
                        stdout=f,stderr=subprocess.STDOUT,timeout=900)
                if checked.returncode:
                    rows.append({'candidate':candidate,'stage':'operators','qualified':False,'returncode':checked.returncode})
                    write(summary,{'controller':record(Path(__file__)),'rows':rows});continue
            for condition in ('c00','c21'):
                if time.time()+900+1200>deadline: raise RuntimeError('Reserve recovery window')
                attempt=f'development-{condition}-{candidate}-r1-001'
                assert not (RUN/'artifacts/attempts'/attempt).exists()
                started=time.monotonic()
                with (folder/(attempt+'.log')).open('x') as f:
                    child=subprocess.run([sys.executable,'-u',str(RUN/'02_benchmark.py'),'--candidate',candidate,
                        '--condition',condition,'--replicate','1','--attempt',attempt,'--development'],
                        stdout=f,stderr=subprocess.STDOUT,timeout=900)
                path=RUN/'artifacts/attempts'/attempt/'result.json'
                result=read(path) if path.exists() else {}
                row={'candidate':candidate,'condition':condition,'attempt':attempt,'returncode':child.returncode,
                     'qualified':result.get('qualified',False),'loss':result.get('loss'),
                     'timing':result.get('timing'),'elapsed_seconds':time.monotonic()-started}
                rows.append(row);write(summary,{'controller':record(Path(__file__)),'rows':rows})
                print(row,flush=True)
    finally:
        lock.unlink()


if __name__=='__main__':main(float(sys.argv[1]))
