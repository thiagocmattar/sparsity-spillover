"""Detached run-local process group under the approved absolute deadline."""
import argparse,shlex,json
from pathlib import Path
from transport import connect,execute

p=argparse.ArgumentParser();p.add_argument('script');p.add_argument('attempt');p.add_argument('extra',nargs=argparse.REMAINDER);a=p.parse_args()
assert a.script.endswith('.py') and '/' not in a.script
assert a.attempt.replace('-','').isalnum()
run=Path(__file__).resolve().parent.parent
q=shlex.quote
body='\n'.join([
 'set -eu', 'cd /workspace/run050',
 'export RUN050_BASE=/workspace/run049',
 'export PATH=/workspace/run049/runtime/venv/bin:/usr/local/cuda/bin:$PATH',
 'export CUDA_HOME=/usr/local/cuda TORCH_EXTENSIONS_DIR=/workspace/run049/runtime/extensions MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=12.0',
 'remaining=$((1790116948 - $(date +%s) - 1200))',
 'test "$remaining" -gt 0',
 'set +e',f'timeout --signal=TERM --kill-after=30 "$remaining" python -u {q(a.script)} --attempt {q(a.attempt)} '+shlex.join(a.extra),
 'code=$?',f'echo "$code" > /workspace/run050-control/{a.attempt}.exit','exit "$code"'])
# Parse rather than trust a hand-converted timestamp.
from datetime import datetime
deadline=int(datetime.fromisoformat('2026-09-22T22:42:28+00:00').timestamp())
body=body.replace('1790116948',str(deadline))
client=connect()
try:
 s=client.open_sftp();remote=f'/workspace/run050-control/{a.attempt}.sh'
 with s.file(remote,'w') as f:f.write(body+'\n')
 s.close()
 cmd=f'nohup setsid bash {q(remote)} > /workspace/run050-control/{a.attempt}.log 2>&1 < /dev/null & echo $! > /workspace/run050-control/{a.attempt}.pid'
 print(execute(client,cmd))
 print(execute(client,f'cat /workspace/run050-control/{a.attempt}.pid'))
 (run/'prelaunch'/f'launch-{a.attempt}.json').write_text(json.dumps({'script':a.script,'attempt':a.attempt,'extra':a.extra,'deadline_epoch':deadline,'command':cmd},indent=2)+'\n')
finally:client.close()
