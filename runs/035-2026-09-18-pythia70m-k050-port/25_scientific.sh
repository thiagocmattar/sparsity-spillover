#!/usr/bin/env bash
set -euo pipefail
cd /workspace/run035
while [ ! -f runtime/qualify-002.exit ]; do sleep 15; done
test "$(cat runtime/qualify-002.exit)" = 0
test -f runtime/setup-complete
runtime/venv/bin/python - <<'PY'
import hashlib,json
from pathlib import Path
r=Path('.')
summary=json.loads((r/'artifacts/qualify/summary-002.json').read_text())
assert len(summary)==6 and all(x['qualified'] and x['status']=='complete' for x in summary)
sources={p.as_posix():{'path':p.as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
         for p in sorted(Path('kernel').rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
for row in summary:
    result=json.loads((r/'artifacts/attempts'/row['attempt']/'result.json').read_text())
    assert result['candidate']=='k050-70m-v2'
    # Compare content identities; record() stores the same run-relative path.
    assert result['port_sources']==sources
with (r/'artifacts/frozen-final-port.json').open('x') as f:
    json.dump({'candidate':'k050-70m-v2','sources':sources,'qualified_preflight':summary},f,indent=2)
PY
bash 05_execute.sh scientific 1789762688.225 001
