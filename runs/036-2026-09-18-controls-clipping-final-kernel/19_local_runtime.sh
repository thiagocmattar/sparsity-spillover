#!/usr/bin/env bash
# Infrastructure-only relocation; retain persistent evidence and identical packages.
set -euo pipefail
cd /workspace/run036
destination=/opt/run036-venv
test ! -e "$destination"
cp -a runtime/venv "$destination"
diff -qr runtime/venv "$destination" > runtime/local-runtime-diff.txt
test ! -s runtime/local-runtime-diff.txt
"$destination/bin/python" -c 'import sys; print(sys.executable, sys.prefix)'
test -L runtime/venv/bin/python
mv runtime/venv/bin/python runtime/venv/bin/python.network-original
printf '%s\n' '#!/bin/sh' 'exec /opt/run036-venv/bin/python "$@"' > runtime/venv/bin/python
chmod 755 runtime/venv/bin/python
python3 - <<'PY'
import json,time
from pathlib import Path
Path('runtime/local-runtime.json').write_text(json.dumps({
 'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
 'source':'/workspace/run036/runtime/venv','destination':'/opt/run036-venv',
 'verification':'cp -a followed by diff -qr: every file byte-identical before wrapper activation',
 'launcher':'Persistent python symlink retained as python.network-original; wrapper execs verified local copy.',
 'scientific_inputs_changed':False},indent=2)+'\n')
PY
