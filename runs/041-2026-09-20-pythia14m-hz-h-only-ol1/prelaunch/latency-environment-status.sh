python3 - <<'PY'
from pathlib import Path
import json
p=Path('/workspace/run041-latency/runtime')
print(json.dumps({'environment_ready':(p/'environment-ready').exists(),'exit':(p/'environment-setup.exit').read_text() if (p/'environment-setup.exit').exists() else None,'last_log':(p/'environment-setup.log').read_text(errors='replace').splitlines()[-8:]}))
PY
