#!/usr/bin/env bash
set -euo pipefail
run026_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
run026_setup_pid="${1:?verified setup PID required}"
while ps -p "$run026_setup_pid" -o stat= | grep -q '^[[:space:]]*[^Z]'; do sleep 15; done
"$run026_dir/runtime/venv/bin/python" -c 'import torch, transformers, numpy; assert torch.__version__.split("+")[0]=="2.11.0"; assert transformers.__version__=="5.12.1"; assert numpy.__version__=="2.5.0"'
bash "$run026_dir/08_gpu_calibrate.sh"
