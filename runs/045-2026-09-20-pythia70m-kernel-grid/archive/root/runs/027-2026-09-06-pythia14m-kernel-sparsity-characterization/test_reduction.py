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


def test_exact_counts_reconcile_with_occupancy():
    diagnostic={'per_site_layer':[],'active_features_per_row':{},'eligible_zero_products':0,
        'eligible_products':338*2048*6*640*128,'eligible_zero_fraction':0.}
    for site in ['a','m','h','z']:
        width=512 if site=='h' else 128
        for layer in range(6):
            name=f'{site}.layer_{layer}'
            diagnostic['per_site_layer'].append({'name':name,'total':692224*width,'exact_zero_count':0})
            if site in ['h','z']:
                diagnostic['active_features_per_row'][name]=[0]*width+[692224]
    reducer.audit_diagnostics(diagnostic)
    diagnostic['active_features_per_row']['h.layer_0'][0]=1
    with pytest.raises(ValueError,match='histogram'):reducer.audit_diagnostics(diagnostic)
    diagnostic['active_features_per_row']['h.layer_0'][0]=0
    diagnostic['eligible_zero_products']=1
    with pytest.raises(ValueError,match='counts disagree'):reducer.audit_diagnostics(diagnostic)
