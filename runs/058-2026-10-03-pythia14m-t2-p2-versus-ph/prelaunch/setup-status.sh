python3 - <<'PY'
from pathlib import Path
import json
control=Path('/workspace/run058-control')
print(json.dumps({'files':{p.name:p.stat().st_size for p in control.glob('*') if p.is_file()},
    'source_bundle_bytes':Path('/workspace/run058-source.bundle').stat().st_size if Path('/workspace/run058-source.bundle').exists() else None}))
for name in ('pipeline.log','cache-build.log','guard.log','pipeline.exit'):
    path=control/name
    if path.exists():
        print(name)
        print('\n'.join(path.read_text(errors='replace').splitlines()[-8:]))
PY
