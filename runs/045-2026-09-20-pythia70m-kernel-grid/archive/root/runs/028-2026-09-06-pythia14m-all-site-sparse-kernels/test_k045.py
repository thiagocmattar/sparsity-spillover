"""The M8 variant accounts for its padded M16 tensor instructions explicitly."""
import importlib.util
from pathlib import Path
import sys
import numpy as np

RUN=Path(__file__).resolve().parent
sys.path.insert(0,str(RUN))
spec=importlib.util.spec_from_file_location('k045_accounting',RUN/'92_k045_joint_probe.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


def test_padding_and_short_row_substitution():
    x=np.zeros((32,512),dtype=np.float32)
    assert probe.accounting(x)==(0,2048,0)
    x[:,0]=1;x[:,37]=-2
    assert probe.accounting(x)==(0,2048,8192)
    assert probe.accounting(x,skip=False)==(2048,0,0)
    assert probe.accounting(x,hybrid=False)==(128,1920,0)


def test_mixed_fallback():
    x=np.zeros((16,128),dtype=np.float32);x[:,0]=1;x[0,[16,32]]=2
    assert probe.accounting(x)==(48,208,15*128)
