"""Execute the unchanged measurements with unambiguous CUDA binding names."""
from functools import lru_cache
import runpy,sys
import bootstrap
import primitives
from support import RUN,sha

@lru_cache(None)
def extension_v2():
    from torch.utils.cpp_extension import load
    path=RUN/'fused_sparse_v2.cu'
    return load(name='r052_sparse_v2_'+sha(path)[:12],sources=[str(path)],
                extra_cuda_cflags=['-O3','-lineinfo','--ptxas-options=-v'],verbose=True)

primitives.extension=extension_v2
script=sys.argv.pop(1)
assert script in ('03_operator_checks.py','04_screen.py','06_benchmark.py','08_diagnostics.py')
sys.argv[0]=str(RUN/script)
runpy.run_path(str(RUN/script),run_name='__main__')
