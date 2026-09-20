set -eu
control=/workspace/run044-control
root=/workspace/run044-latency
test ! -e "$control/compile-003.log"
python3 - <<'PY'
from pathlib import Path
import json,os,signal,time
control=Path('/workspace/run044-control')
pid=int((control/'pipeline.pid').read_text())
command=Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\0',' ')
assert '00_precompile.py' in command and os.getpgid(pid)==pid
(control/'compile-002-interruption.json').write_text(json.dumps({'pid':pid,'epoch':time.time(),
    'reason':'Restart with the exact TORCH_EXTENSIONS_DIR and environment used by 05_execute.sh; preserve compile002 log.'},indent=2)+'\n')
os.killpg(pid,signal.SIGTERM)
PY
nohup setsid bash -c 'export PATH=/workspace/run044-latency/runtime/venv/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda TORCH_EXTENSIONS_DIR=/workspace/run044-latency/runtime/extensions TRITON_CACHE_DIR=/workspace/run044-latency/runtime/triton MAX_JOBS=2 HF_HUB_OFFLINE=1; unset CUBLAS_WORKSPACE_CONFIG; cd /workspace/run044-latency; python -u 00_precompile.py; code=$?; printf "%s\n" "$code" > /workspace/run044-control/compile-003.exit' > "$control/compile-003.log" 2>&1 < /dev/null &
echo "$!" > "$control/pipeline.pid"
echo 'Compile retry003 launched using the final benchmark environment and cache paths'
