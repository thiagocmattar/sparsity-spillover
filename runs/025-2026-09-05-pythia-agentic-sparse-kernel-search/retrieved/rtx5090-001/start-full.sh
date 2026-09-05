#!/usr/bin/env bash
set -euo pipefail
run025_root=/workspace/sparsity-spillover
run025_dir="$run025_root/runs/025-2026-09-05-pythia-agentic-sparse-kernel-search"
run025_venv="$run025_dir/artifacts/runtime/venv-pythia"
export PATH="$run025_venv/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
[[ "$(cat "$run025_dir/artifacts/memcheck-002.exit-code")" == 0 ]]
grep -q 'ERROR SUMMARY: 0 errors' "$run025_dir/artifacts/memcheck-002.log"
printf '%s  %s\n' 1a2a10488f29f52c202affa906b0e7aca29453b2e97132e02af26ba7dae710f1 /workspace/run025-calibration.tar | sha256sum --check
tar --skip-old-files -xf /workspace/run025-calibration.tar -C "$run025_root"
"$run025_venv/bin/python" "$run025_dir/02_package.py" --verify "$run025_dir/prelaunch/transfer-calibration.json"
run025_seconds=$(( $(date -u -d '2026-09-05 17:06:20 UTC' +%s) - $(date -u +%s) - 600 ))
(( run025_seconds > 0 ))
(( run025_seconds <= 5400 )) || run025_seconds=5400
bash "$run025_dir/03_start_calibration.sh" "$run025_venv/bin/python" rtx5090-calibration-001 "$run025_seconds"
