python3 - <<'PY'
from pathlib import Path
import json
run=Path('/workspace/sparsity-spillover/runs/058-2026-10-03-pythia14m-t2-p2-versus-ph')
control=Path('/workspace/run058-control')
archive=run/'transfer/training-001.tar.gz'
receipt=run/'transfer/training-receipt-001.json'
print(json.dumps({'archive_bytes':archive.stat().st_size if archive.exists() else None,
    'receipt':json.loads(receipt.read_text()) if receipt.exists() else None,
    'exit':(control/'seal.exit').read_text().strip() if (control/'seal.exit').exists() else None}))
log=control/'seal.log'
if log.exists():print('\n'.join(log.read_text(errors='replace').splitlines()[-5:]))
PY
