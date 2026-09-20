python3 - <<'PY'
from pathlib import Path
import json
p=Path('/workspace/run043-latency/runtime')
c=Path('/workspace/run043-control')
print(json.dumps({'environment_ready':(p/'environment-ready').exists(),'exit':(c/'environment-setup.exit').read_text() if (c/'environment-setup.exit').exists() else None,'last_log':(c/'environment-setup.log').read_text(errors='replace').splitlines()[-8:]}))
PY
