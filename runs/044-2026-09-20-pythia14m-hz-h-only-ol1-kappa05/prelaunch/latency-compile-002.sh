set -eu
control=/workspace/run044-control
root=/workspace/run044-latency
test -e "$root/runtime/environment-ready"
test ! -e "$control/compile-002.log"
nohup setsid bash -c 'export PATH=/workspace/run044-latency/runtime/venv/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda; cd /workspace/run044-latency; python 00_precompile.py; code=$?; printf "%s\n" "$code" > /workspace/run044-control/compile-002.exit' > "$control/compile-002.log" 2>&1 < /dev/null &
echo "$!" > "$control/pipeline.pid"
echo 'Compile retry002 launched with the existing frozen environment and deadline'
