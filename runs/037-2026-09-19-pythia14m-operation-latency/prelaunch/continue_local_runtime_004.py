"""Resume the unchanged scientific matrix from a local runtime after all smokes."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

RUN=Path('/workspace/run037')
OUT=RUN/'artifacts/infrastructure/004-direct-local-runtime'
DEADLINE=1789829330.832

if __name__=='__main__':
    command=Path('/proc/373/cmdline').read_bytes()
    assert b'05_execute.sh' in command
    os.kill(373,signal.SIGSTOP)
    print('Original shell held; smoke controller and current child run unchanged',flush=True)
    replaced=False
    try:
        while True:
            status=Path('/proc/606/status')
            if not status.exists() or '\nState:\tZ' in status.read_text():break
            if time.time()>DEADLINE-900:raise RuntimeError('Insufficient remaining launch envelope')
            time.sleep(5)
        rows=json.loads((RUN/'artifacts/smoke/summary-001.json').read_text())
        assert len(rows)==10 and all(r['qualified'] for r in rows)
        assert (RUN/'artifacts/infrastructure/003-local-runtime/metadata-restoration.json').exists()
        assert not (RUN/'artifacts/controller.lock').exists()
        # Only the idle shell is replaced; every GPU child has completed.
        os.kill(373,signal.SIGKILL)
        replaced=True
        env=os.environ.copy()
        env.update(PATH='/tmp/run037-local/venv/bin:/usr/local/cuda/bin:'+env['PATH'],
                   CUDA_HOME='/usr/local/cuda',MAX_JOBS='2',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                   TORCH_EXTENSIONS_DIR='/workspace/run037-extensions',PYTHONUNBUFFERED='1',
                   PYTHONPATH=str(OUT/'bridge'))
        python='/tmp/run037-local/venv/bin/python'
        commands=[['03_execute.py','--phase','scientific','--deadline-epoch',str(DEADLINE)],
                  ['07_reduce.py'],['08_collect.py']]
        for args in commands:
            print(json.dumps({'running':args,'epoch':time.time()}),flush=True)
            subprocess.run([python,*args],cwd=RUN,env=env,check=True)
        (OUT/'execution.exit').write_text('0\n')
    except BaseException:
        (OUT/'execution.exit').write_text('1\n')
        raise
    finally:
        if not replaced and Path('/proc/373').exists():os.kill(373,signal.SIGCONT)
