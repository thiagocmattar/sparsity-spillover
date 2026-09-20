python3 - <<'PY'
from pathlib import Path
import json
root=Path('/workspace/run041-latency/artifacts/attempts')
for p in sorted(root.glob('*/quality.json')):
 q=json.loads(p.read_text())
 print(json.dumps({'attempt':p.parent.name,'pass':q['pass'],'loss_delta':q['loss_delta'],'max_abs':max(g['max_abs'] for g in q['gates']['candidate_graph']),'max_relative_l2':max(g['relative_l2'] for g in q['gates']['candidate_graph'])}))
PY
