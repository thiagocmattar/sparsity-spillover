python3 - <<'PY'
from pathlib import Path
p=Path('/workspace/run041-latency/artifacts/smoke/smoke-c00-r1-001.log')
with p.open('rb') as f:
 f.seek(max(0,p.stat().st_size-2300));print(f.read().decode(errors='replace'))
PY
