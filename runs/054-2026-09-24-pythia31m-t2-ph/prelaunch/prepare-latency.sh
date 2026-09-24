set -euo pipefail
cd /workspace/sparsity-spillover
export PYTHONPATH=$PWD/src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
/workspace/run054-venv/bin/python - <<'PY'
import sys,pathlib,torch,transformers
r=pathlib.Path('runs/054-2026-09-24-pythia31m-t2-ph');sys.path.insert(0,str(r))
from model_factory import build_pinned_run054_model
from initialization_artifact import load_pinned_initialization
from run_config import load_config,condition_specs,resolved_condition_config
c=load_config();c=resolved_condition_config(c,condition_specs(c)[0])
m=build_pinned_run054_model(c['model'],device=torch.device('cpu'),torch=torch,auto_model=transformers.AutoModelForCausalLM)
load_pinned_initialization(m,torch=torch);m.save_pretrained(r/'prelaunch/kernel-initial-base')
PY
/workspace/run054-venv/bin/python runs/054-2026-09-24-pythia31m-t2-ph/latency/02_component_check.py
/workspace/run054-venv/bin/python runs/054-2026-09-24-pythia31m-t2-ph/latency/01_benchmark.py --checkpoint runs/054-2026-09-24-pythia31m-t2-ph/prelaunch/kernel-initial-base --condition calibration-base --attempt remote-initial-base-001 --local-calibration --smoke
touch /workspace/run054-control/kernel-ready
