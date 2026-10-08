#!/usr/bin/env bash
set -euo pipefail
run=/workspace/run067/runs/067-2026-10-08-pythia14m-sparse-vocabulary-kernel
python - <<'PY'
import hashlib,json,subprocess,tarfile
from pathlib import Path
run=Path('/workspace/run067/runs/067-2026-10-08-pythia14m-sparse-vocabulary-kernel')
assert (run/'ops/exit-001.txt').read_text().strip()=='0'
seal=json.loads((run/'transfer/seal-001.json').read_text())
archive=run/seal['archive']['path'];payload=archive.read_bytes()
assert len(payload)==seal['archive']['bytes'] and hashlib.sha256(payload).hexdigest()==seal['archive']['sha256']
hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.total,driver_version','--format=csv'],text=True)
hardware+=subprocess.check_output(['/usr/local/cuda/bin/nvcc','--version'],text=True)
(run/'ops/hardware-001.txt').write_text(hardware)
paths=[archive,run/'transfer/seal-001.json']+sorted(p for p in (run/'ops').iterdir() if p.is_file())
rows=[{'path':p.relative_to(run).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]
inventory=run/'ops/delivery-inventory-001.json';inventory.write_text(json.dumps(rows,indent=2)+'\n')
destination=Path('/workspace/run067-delivery-001.tar.gz')
with tarfile.open(destination,'w:gz') as handle:
 for p in paths+[inventory]:handle.add(p,arcname=p.relative_to(run).as_posix(),recursive=False)
payload=destination.read_bytes();receipt={'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest(),'members':len(rows)+1}
(run/'ops/delivery-seal-001.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(receipt,flush=True)
PY
setsid runpodctl send /workspace/run067-delivery-001.tar.gz > /workspace/run067-return-private.out 2> /workspace/run067-return-private.err < /dev/null &
printf 'RETURN_SENDER %s\n' "$!"
