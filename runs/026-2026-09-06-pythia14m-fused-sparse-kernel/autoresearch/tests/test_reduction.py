import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
spec = importlib.util.spec_from_file_location('run026_reduction', HERE/'reduce_final.py')
reduction = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reduction)


def test_pairing_and_geometric_estimand():
    rows = [{'input_index':0,'repeat':r,'mode':m,'host_ms':v}
            for r, values in enumerate([(2.,1.),(8.,1.)])
            for m,v in zip(['native','candidate'],values)]
    cube, times = reduction.paired_cube([{'samples':rows}], inputs=1, passes=2)
    assert np.exp(cube.mean()) == pytest.approx(4.)
    assert reduction.crossed_interval(cube,draws=20) == pytest.approx([4.,4.])
    assert len(times['native']) == 2


def test_missing_duplicate_and_nonpositive_rejected():
    row = {'input_index':0,'repeat':0,'mode':'native','host_ms':1.}
    for rows in ([row], [row,row], [dict(row,host_ms=0)]):
        with pytest.raises(ValueError):
            reduction.paired_cube([{'samples':rows}], inputs=1, passes=1)


def test_crossed_interval_retains_cluster_variation():
    cube = np.log(np.array([[[1.,1.],[1.,1.]],[[4.,4.],[4.,4.]]]))
    low, high = reduction.crossed_interval(cube,draws=500)
    assert low == pytest.approx(1.)
    assert high == pytest.approx(4.)
