import importlib.util
from pathlib import Path
import sys
import pytest
import torch

RUN=Path(__file__).resolve().parent
sys.path.insert(0,str(RUN))
spec=importlib.util.spec_from_file_location('k045_counter_v2',RUN/'98_k045_joint_probe_v2.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


@pytest.mark.parametrize('shape',[(3,8,6),(3,2,2,6)])
def test_reduction_keeps_named_counter_axis(shape):
    stats=torch.zeros(shape,dtype=torch.int64)
    flat=stats.reshape(-1,6);flat[0]=torch.arange(6);flat[-1]=2*torch.arange(6)
    assert probe.sum_work_counts(stats)==[0,3,6,9,12,15]


def test_reduction_rejects_missing_field_axis():
    with pytest.raises(ValueError):probe.sum_work_counts(torch.zeros(8,5))
