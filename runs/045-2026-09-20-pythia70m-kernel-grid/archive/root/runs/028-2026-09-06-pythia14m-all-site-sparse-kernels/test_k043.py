"""CPU audit of the hybrid accounting reference, not GPU qualification."""
import importlib.util
from pathlib import Path
import sys
import numpy as np

RUN=Path(__file__).resolve().parent
sys.path.insert(0,str(RUN))
spec=importlib.util.spec_from_file_location('k043_accounting',RUN/'86_k043_joint_probe.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


def test_short_rows_and_dense_control():
    x=np.zeros((32,512),dtype=np.float32)
    assert probe.accounting(x)==(0,1024,0)
    x[:,0]=1;x[:,37]=-2
    assert probe.accounting(x)==(0,1024,8192)
    assert probe.accounting(x,skip=False)==(1024,0,0)
    assert probe.accounting(x,hybrid=False)==(64,960,0)


def test_mixed_fallback_and_unsafe_range():
    x=np.zeros((16,128),dtype=np.float32);x[:,0]=1
    x[0,[16,32]]=2
    assert probe.accounting(x)==(48,80,15*128)
    x[:]=0;x[:,0]=2.**-60
    assert probe.accounting(x)==(16,112,0)
