import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parent))
spec=importlib.util.spec_from_file_location('run027_reduce',Path(__file__).with_name('10_reduce.py'))
reducer=importlib.util.module_from_spec(spec);spec.loader.exec_module(reducer)


def test_geometric_pairing_and_rejections():
    samples=[{'mode':mode,'host_ms':value,'input_index':0,'repeat':i}
        for i,pair in enumerate([(2.,1.),(8.,1.)]) for mode,value in zip(['native','sparse'],pair)]
    cube,_=reducer.ratio_cube([{'samples':samples}],'native','sparse',n_inputs=1,passes=2)
    assert np.exp(cube.mean())==pytest.approx(4.)
    assert reducer.interval(cube,draws=10)==pytest.approx([4.,4.])
    for bad in [samples[:-1],samples+[samples[0]]]:
        with pytest.raises(ValueError):reducer.ratio_cube([{'samples':bad}],'native','sparse',n_inputs=1,passes=2)


def test_process_clusters_not_individual_repeats():
    cube=np.log(np.array([[[1.]*7]*4,[[4.]*7]*4]))
    assert reducer.interval(cube,draws=1000)==pytest.approx([1.,4.])
