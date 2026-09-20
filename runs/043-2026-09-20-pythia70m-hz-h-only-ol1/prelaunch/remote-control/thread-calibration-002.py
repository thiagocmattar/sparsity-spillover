import importlib.util,json
from pathlib import Path
root=Path('/workspace/sparsity-spillover/runs/043-2026-09-20-pythia70m-hz-h-only-ol1')
spec=importlib.util.spec_from_file_location('calibration',root/'05_remote_preflight.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.RUN_DIR=root/'prelaunch/thread-calibration-002'
module.main()
