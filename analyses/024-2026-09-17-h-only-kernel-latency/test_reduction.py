"""Check paired alignment, count coverage and estimand, without fabricated figures."""
import importlib.util
import math
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('reduce024',Path(__file__).with_name('01_reduce.py'))
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def samples():
    return [{'repeat':r,'input_index':i,'mode':mode,'output_shape':[1,2048,50304],
             'host_ms': (100. if i==0 else 1.) * (2. if mode=='native_graph' else 1.)}
            for r in range(7) for i in range(64) for mode in ['candidate_graph','native_graph']]


def test_align_pairs_independent_of_mode_order_and_use_geometric_mean():
    n,c,pairs=mod.paired_values(list(reversed(samples())))
    assert len(pairs)==448 and math.isclose(mod.gm(pairs),2.)
    assert math.isclose(mod.gm(n)/mod.gm(c),2.)
    assert mod.gm(c)>1.0  # Distinct from the process median of 1 ms.


def test_reject_missing_or_duplicated_raw_pair():
    with pytest.raises(AssertionError):mod.paired_values(samples()[:-1])
    with pytest.raises(AssertionError):mod.paired_values(samples()+samples()[:1])


def test_reject_nonpositive_or_nonfinite_times():
    for value in [0.,-1.,float('nan'),float('inf')]:
        with pytest.raises(AssertionError):mod.gm([1.,value])
