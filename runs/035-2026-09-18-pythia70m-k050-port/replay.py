"""Same frozen measurement scaffold; separately identified K050-derived port."""
import sys
from io_utils import RUN,ARCHIVE,archived_run,module

R25,R26,R27,R28=[archived_run(n) for n in [25,26,27,28]]
sys.path[:0]=[str(R28),str(R27),str(R25),str(ARCHIVE/'src')]
common=module('common',R28/'common.py')
dense=common.dense
scaffold=module('run035_frozen_graph',R28/'45_graph_forward.py')

def final_row(name,catalog):return next(r for r in catalog if r['id']==name)

def install(model,row):
    if row['candidate']!='k050-70m-v1':raise ValueError('Unknown frozen candidate')
    candidate=module('run035_candidate',RUN/'kernel/candidate.py')
    return candidate.install(model,**row['settings'])
