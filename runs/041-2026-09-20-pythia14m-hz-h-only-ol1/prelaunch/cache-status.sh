python3 - <<'PY'
from pathlib import Path
import json,shutil
p=Path('/workspace/run041-control/cache.log')
lines=p.read_text(errors='replace').splitlines()
rows=[]
for line in lines:
 try:
  row=json.loads(line)
  if row.get('event') in ['tokenization_progress','cache_published']:rows.append(row)
 except ValueError:pass
print(json.dumps({'last_progress':rows[-1] if rows else None,'free_disk_GiB':shutil.disk_usage('/workspace').free/1024**3,'pipeline_exit':Path('/workspace/run041-control/pipeline.exit').read_text() if Path('/workspace/run041-control/pipeline.exit').exists() else None}))
PY
