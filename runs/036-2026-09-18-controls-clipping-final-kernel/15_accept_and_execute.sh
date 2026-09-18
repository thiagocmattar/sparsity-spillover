#!/usr/bin/env bash
set -euo pipefail
cd /workspace/run036
while [ ! -f runtime/relay-bundle-001.exit ]; do sleep 5; done
test "$(cat runtime/relay-bundle-001.exit)" = 0
python3 - <<'PY'
import json,hashlib
from pathlib import Path
p=Path('relay-bundle-001/input-001.tar');r=json.loads(Path('input-receipt.json').read_text())
assert p.stat().st_size==r['bytes']
with p.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==r['sha256']
print('INPUT_ARCHIVE_HASH_VERIFIED',flush=True)
PY
tar -xf relay-bundle-001/input-001.tar
touch runtime/inputs-ready
while [ ! -f runtime/setup.exit ]; do sleep 5; done
test "$(cat runtime/setup.exit)" = 0
test -f runtime/setup-complete
bash 05_execute.sh "$1"
