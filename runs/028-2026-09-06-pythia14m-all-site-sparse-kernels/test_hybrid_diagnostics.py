import importlib.util
from pathlib import Path
import sys
import numpy as np
import torch

RUN=Path(__file__).resolve().parent
sys.path.insert(0,str(RUN))
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,RUN/path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
diagnostic=load('hybrid_diag_test','115_hybrid_diagnostics.py')
reference=load('hybrid_ref_test','109_k049_joint_probe.py')


def test_counting_matches_independent_numpy_reference():
    rng=np.random.default_rng(2851)
    for width in [128,512]:
        for occupancy in [0.,.001,.01,.5,1.]:
            x=rng.normal(size=(32,width)).astype(np.float32)
            x[rng.random(x.shape)>occupancy]=0
            for fast in [True,False]:
                for skip in [True,False]:
                    assert diagnostic.hybrid_counts(torch.tensor(x),fast,skip)==list(reference.accounting(x,fast,skip))


def test_padding_simt_and_unsafe_values():
    x=torch.zeros(16,128)
    assert diagnostic.hybrid_counts(x)==[0,256,0]
    x[:,0]=1
    assert diagnostic.hybrid_counts(x)==[0,256,16*128]
    assert diagnostic.hybrid_counts(x,False)==[32,224,0]
    assert diagnostic.hybrid_counts(x,True,False)==[256,0,0]
    for magnitude in [2.**-60,2.**60]:
        x[:,0]=magnitude
        assert diagnostic.hybrid_counts(x)==[32,224,0]
