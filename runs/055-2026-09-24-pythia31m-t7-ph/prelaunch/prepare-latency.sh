set -euo pipefail
cd /workspace/sparsity-spillover
export PATH=/workspace/run055-venv/bin:$PATH
export PYTHONPATH=$PWD/src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
/workspace/run055-venv/bin/python - <<'PY'
import sys,pathlib,torch,transformers,json
r=pathlib.Path('runs/055-2026-09-24-pythia31m-t7-ph');sys.path.insert(0,str(r))
from model_factory import build_pinned_run055_model
from initialization_artifact import load_pinned_initialization
from run_config import load_config,condition_specs,resolved_condition_config
c=load_config()
for row in condition_specs(c):
    cfg=resolved_condition_config(c,row)
    m=build_pinned_run055_model(cfg['model'],device=torch.device('cpu'),torch=torch,auto_model=transformers.AutoModelForCausalLM)
    load_pinned_initialization(m,torch=torch)
    m.save_pretrained(r/'prelaunch'/('kernel-initial-'+row['id']))
PY
/workspace/run055-venv/bin/python runs/055-2026-09-24-pythia31m-t7-ph/latency/02_component_check.py
for condition in a7-h-ol1-kappa-0 a7-h-ol1-kappa-0p01 a7-h-ol1-kappa-0p05 a7-h-ol1-kappa-0p1 a7-h-ol1-kappa-0p5; do
  /workspace/run055-venv/bin/python runs/055-2026-09-24-pythia31m-t7-ph/latency/01_benchmark.py --checkpoint runs/055-2026-09-24-pythia31m-t7-ph/prelaunch/kernel-initial-$condition --condition calibration-$condition --attempt remote-initial-$condition-001 --local-calibration
done
/workspace/run055-venv/bin/python - <<'PY'
import pathlib,json
r=pathlib.Path('runs/055-2026-09-24-pythia31m-t7-ph/latency/artifacts/attempts')
rows=[json.loads(p.read_text()) for p in r.glob('remote-initial-*-001/manifest.json')]
assert len(rows)==5 and all(x['status']=='completed' and x['qualified'] for x in rows)
PY
touch /workspace/run055-control/kernel-ready
