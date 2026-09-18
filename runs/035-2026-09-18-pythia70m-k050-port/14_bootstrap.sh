#!/usr/bin/env bash
set -euo pipefail
cd /workspace/run035
while [ ! -f runtime/relay-bootstrap-002.exit ]; do sleep 5; done
test "$(cat runtime/relay-bootstrap-002.exit)" = 0
python3 - <<'PY'
import json,hashlib
from pathlib import Path
p=Path('relay-bootstrap-002/bootstrap-002.tar');r=json.loads(Path('bootstrap-receipt.json').read_text())
assert p.stat().st_size==r['bytes']
with p.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==r['sha256']
print('BOOTSTRAP_HASH_VERIFIED',flush=True)
PY
tar -xf relay-bootstrap-002/bootstrap-002.tar
while [ ! -s runtime/nvidia-smi-initial.txt ]; do
  if [ -f runtime/setup.exit ]; then echo 'SETUP_FAILED'; exit 1; fi
  sleep 5
done
export PATH=/workspace/run035/runtime/venv/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
export TORCH_EXTENSIONS_DIR=/workspace/run035/runtime/extensions MAX_JOBS=2
mkdir -p artifacts
runtime/venv/bin/python -u 10_test_operators.py
