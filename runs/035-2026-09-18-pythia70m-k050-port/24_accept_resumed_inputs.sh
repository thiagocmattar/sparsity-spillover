#!/usr/bin/env bash
set -euo pipefail
cd /workspace/run035
while [ ! -f runtime/relay-bundle-002.exit ]; do sleep 5; done
test "$(cat runtime/relay-bundle-002.exit)" = 0
python3 - <<'PY'
import json,hashlib
from pathlib import Path
p=Path('relay-bundle-001/input-001.tar');r=json.loads(Path('input-receipt.json').read_text())
assert p.stat().st_size==r['bytes']
with p.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==r['sha256']
print('FULL_INPUT_ARCHIVE_HASH_VERIFIED',flush=True)
PY
tar -xf relay-bundle-001/input-001.tar --wildcards 'inputs/checkpoints/*'
touch runtime/inputs-ready
