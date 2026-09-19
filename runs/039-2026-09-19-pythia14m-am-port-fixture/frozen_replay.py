"""Installation-only compatibility layer over byte-identical archived kernels."""
import sys
from io_utils import ARCHIVE, archived_run, module

R25, R26, R27, R28 = [archived_run(n) for n in [25,26,27,28]]
sys.path[:0] = [str(R28),str(R27),str(R25),str(ARCHIVE/'src')]
common = module('common',R28/'common.py')
dense = common.dense
scaffold = module('run029_frozen_graph',R28/'45_graph_forward.py')


def install(model, row):
    from sparsity_research.pythia import topology_metadata
    name = row['candidate']
    settings = row['settings']
    topology = topology_metadata(model)
    sites = set(topology['active_sites']) & {'a','m','h','z'}
    if name=='p0' or int(name[1:])<=16:
        source = R25/'p0.py' if name=='p0' else R25/f'autoresearch/candidates/{name}/candidate.py'
        candidate = module('run029_'+row['id'].replace('-','_'),source)
        kwargs = {k:v for k,v in settings.items() if k!='sites'}
        adapter = candidate.Adapter(model,**kwargs)
        adapter.set_mode(name)
        # The original Run025 model probes used this same topology-active filter.
        for parent, attribute, original, wrapped, site in adapter.entries:
            if site not in sites:
                setattr(parent,attribute,original)
        return {'topology':topology,'active_linear_sites':sorted(sites),
                'implementation':adapter.coverage(),'settings':settings}
    n = int(name[1:])
    if n in [17,18]:
        if not {'a','m','h','z'} <= set(topology['active_sites']):
            raise ValueError('UNSUPPORTED: historical K017/K018 requires existing a/m/h/z gates')
        candidate = module('run029_'+name,R26/f'autoresearch/candidates/{name}/candidate.py')
        if n==17:
            candidate.Adapter(model,**settings).set_mode(True)
        else:
            candidate.Adapter(model).install()
        return {'topology':topology,'settings':settings}
    original = module('run029_frozen_k019_port',R27/'adapter.py')
    original.install(model,'sparse')
    if n==19:
        return {'topology':topology,'composition':'Run027 K019+K018 compatibility port'}
    candidate = module('run029_'+name,R28/f'candidates/{name}/candidate.py')
    return candidate.install(model,**settings)


def final_row(name, catalog):
    if name in {'k050-no-skip','k050-attention-dense'}:
        row = dict(next(r for r in catalog if r['id']=='k050'))
        row['settings'] = {**row['settings'],'skip':False,'projection_skip':name!='k050-no-skip'}
        row['id'] = name
        return row
    return next(r for r in catalog if r['id']==name)
