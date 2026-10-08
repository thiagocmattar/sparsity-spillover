"""Select frozen implementations, each architecture in a fresh process."""
import sys
from runio import DEPS, module, read

def setup(size):
    source=DEPS/'runs'/('045-2026-09-20-pythia70m-kernel-grid' if size=='31m' else '049-2026-09-22-pythia70m-short-row-limits')
    sys.path.insert(0,str(source))
    import replay
    if size=='31m':
        port=module('run065_port31',DEPS/'runs/054-2026-09-24-pythia31m-t2-ph/latency/port.py')
        def install(model,mode):
            meta=port.install(model)
            if mode=='no_skip':
                for layer in model.gpt_neox.layers:layer._run026_joint.skip=False
            return {**meta,'mode':mode}
    else:
        run=DEPS/'runs/051-2026-09-22-pythia70m-frozen-kernel-grid'
        sys.path.insert(0,str(run))
        import adapter
        selection=read(run/'provenance/selection-final.json')
        def install(model,mode):
            policy=selection['policies']['sparse_c'] if mode=='sparse' else selection['ablations']['sparse_c_no_skip']
            return {'identity':'Run051 sparse_c','mode':mode,'policy':policy,'layers':adapter.install(model,policy)}
    return replay,install
