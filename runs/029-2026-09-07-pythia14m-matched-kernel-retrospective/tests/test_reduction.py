from pathlib import Path
import sys
import pytest

RUN=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(RUN))
from io_utils import module
reduce=module('run029_reduction_test',RUN/'10_reduce.py')


def test_pair_reduction_and_rejections():
    cfg={'timing_passes':1,'timing_inputs':2}
    rows=[{'repeat':0,'input_index':i,'mode':m,'host_ms':v,'output_shape':[1,2048,50304]}
          for i,pair in enumerate([(2.,1.),(8.,2.)]) for m,v in zip(['native_graph','candidate_graph'],pair)]
    assert reduce.geometric(reduce.ratios(rows,cfg))==pytest.approx(8**.5)
    with pytest.raises(ValueError,match='Duplicate'):reduce.ratios(rows+[rows[0]],cfg)
    with pytest.raises(ValueError,match='Incomplete'):reduce.ratios(rows[:-1],cfg)
    bad=[dict(r) for r in rows];bad[0]['output_shape']=[1,2048,128]
    with pytest.raises(ValueError,match='full-model'):reduce.ratios(bad,cfg)


def test_incumbent_fixed_checkpoint_and_qualification():
    catalog=[{'id':f'k{i:03d}','candidate':f'k{i:03d}','paper_iteration':i,'status':'eligible'} for i in range(1,5)]
    def point(k,s,condition='c30',qualified=True):
        return {'phase':'history','candidate':f'k{k:03d}','condition':condition,'qualified':qualified,'speedup':s}
    points=[point(1,.8),point(2,1.2),point(3,3.,qualified=False),point(3,4.,condition='c01'),point(4,1.1)]
    result=reduce.progress(points,catalog,'c30')
    assert [r['speedup'] for r in result]==[1.,1.,1.2,1.2,1.2]
    assert result[-1]['incumbent']=='k002'


def test_regression_estimated_intercept():
    fit=reduce.ols([0,.1,.2,.3],[.9,1.2,1.5,1.8])
    assert fit['intercept']==pytest.approx(.9)
    assert fit['slope']==pytest.approx(3.)
    assert fit['r_squared']==pytest.approx(1.)
    with pytest.raises(ValueError):reduce.ols([0,0,0],[1,2,3])
