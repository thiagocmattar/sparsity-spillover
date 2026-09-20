"""Run028 fixed inputs and reused, independently retained measurement helpers."""
from pathlib import Path
import importlib.util
import sys

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
R27 = ROOT/'runs/027-2026-09-06-pythia14m-kernel-sparsity-characterization'
sys.path.insert(0, str(R27))
from run027_common import R25, R26, read_json, write_json, record, verify_record, sha256, dense


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def manifest():
    value = read_json(R27/'prelaunch/inputs.json')
    for row in value['inputs'].values():
        verify_record(row)
    return value


def runtime(torch):
    import numpy as np
    import transformers
    import os
    pins = read_json(R25/'config.json')['runtime']
    if (torch.__version__.split('+')[0], transformers.__version__, np.__version__) != (pins['torch'], pins['transformers'], pins['numpy']):
        raise ValueError('Pinned runtime differs')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG'):
        raise ValueError('Use the canonical default workspace')
    torch.manual_seed(2801)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
