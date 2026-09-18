"""Frozen 14M K050 and qualified 70M port, with timed clipping adapters."""
import sys
from io_utils import RUN, ARCHIVE, archived_run, module

R25, R26, R27, R28 = [archived_run(n) for n in (25, 26, 27, 28)]
sys.path[:0] = [str(R28), str(R27), str(R25), str(ARCHIVE/'src')]
common = module('common', R28/'common.py')
dense = common.dense
scaffold = module('run036_graph', R28/'45_graph_forward.py')


def final_row(name, catalog):
    return next(r for r in catalog if r['id'] == name)


def install(model, row):
    if row['candidate'] == 'k050':
        module('run036_adapter14', R27/'adapter.py').install(model, 'sparse')
        candidate = module('run036_k050', R28/'candidates/k050/candidate.py')
    elif row['candidate'] == 'k050-70m-v2':
        candidate = module('run036_port70', RUN/'kernel70/candidate.py')
    else:
        raise ValueError('Only frozen final kernels are in this study')
    return candidate.install(model, **row['settings'])
