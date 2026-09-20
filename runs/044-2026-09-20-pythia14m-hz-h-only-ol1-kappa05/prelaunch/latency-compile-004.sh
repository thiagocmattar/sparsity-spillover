set -eu
control=/workspace/run044-control
root=/workspace/run044-latency
test "$(cat "$control/compile-003.exit")" = 0
test ! -e "$control/compile-004.log"
cat > "$control/compile-004.py" <<'PY'
import json,time
import replay
from io_utils import RUN,module
started=time.monotonic()
for name,path in [
    ('k019',replay.R26/'autoresearch/candidates/k019/candidate.py'),
    ('k035',replay.R28/'candidates/k035/candidate.py'),
]:
    module('run044_precompile_'+name,path).extension()
    print(json.dumps({'compiled':name,'elapsed_seconds':time.monotonic()-started}),flush=True)
(RUN/'runtime/precompile-extra-ready').touch()
PY
nohup setsid bash -c 'export PATH=/workspace/run044-latency/runtime/venv/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda TORCH_EXTENSIONS_DIR=/workspace/run044-latency/runtime/extensions TRITON_CACHE_DIR=/workspace/run044-latency/runtime/triton MAX_JOBS=2 HF_HUB_OFFLINE=1 PYTHONPATH=/workspace/run044-latency; unset CUBLAS_WORKSPACE_CONFIG; cd /workspace/run044-latency; python -u /workspace/run044-control/compile-004.py; code=$?; printf "%s\n" "$code" > /workspace/run044-control/compile-004.exit' > "$control/compile-004.log" 2>&1 < /dev/null &
echo "$!" > "$control/pipeline.pid"
echo 'Compiling the unchanged K019 and K035 dependencies before checkpoint arrival'
