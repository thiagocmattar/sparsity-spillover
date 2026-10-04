python3 - <<'PY'
from pathlib import Path
import json
control=Path('/workspace/run062-control')
run=Path('/workspace/sparsity-spillover/runs/062-2026-10-04-pythia14m-t7-corner-tile-ph')
exit_path=control/'seal-training.exit'
receipt=run/'transfer/training-receipt-001.json'
archive=run/'transfer/training-001.tar.gz'
print(json.dumps({'seal_exit':exit_path.read_text().strip() if exit_path.exists() else None,
 'receipt':json.loads(receipt.read_text()) if receipt.exists() else None,
 'archive_bytes':archive.stat().st_size if archive.exists() else None,
 'log_tail':(control/'seal-training.log').read_text(errors='replace')[-1600:]},indent=2))
PY
