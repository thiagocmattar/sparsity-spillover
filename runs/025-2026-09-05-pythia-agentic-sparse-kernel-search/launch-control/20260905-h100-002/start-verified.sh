#!/usr/bin/env bash
set -euo pipefail
printf '%s  %s\n' a9bed29d27a13652e8901c37946d5f09eaea6ebc5849ea359d12e9f2763f6d9c /workspace/run025-primitive.tar | sha256sum --check
mkdir -p /workspace/sparsity-spillover
tar -xf /workspace/run025-primitive.tar -C /workspace/sparsity-spillover
run025_dir=/workspace/sparsity-spillover/runs/025-2026-09-05-pythia-agentic-sparse-kernel-search
cp /workspace/bootstrap-h100-001.sh "$run025_dir/bootstrap-h100-001.sh"
bash "$run025_dir/bootstrap-h100-001.sh" "${1:?Pass the lease deadline Unix seconds}"
