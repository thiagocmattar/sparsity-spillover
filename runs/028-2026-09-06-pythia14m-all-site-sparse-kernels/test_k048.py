"""Warp-ballot early exit recognizes exactly the same bounded short rows."""
import numpy as np


def ballot_classify(values):
    selected=[]
    for base in range(0,len(values),32):
        mask=sum(1<<lane for lane,value in enumerate(values[base:base+32]) if value!=0)
        if len(selected)+mask.bit_count()>2:return None
        while mask:
            lane=(mask&-mask).bit_length()-1;value=values[base+lane]
            if not 2.**-50<=abs(value)<=2.**50:return None
            selected.append(base+lane);mask&=mask-1
    return selected


def test_positions_and_density():
    rng=np.random.default_rng(2848)
    for width in [128,512]:
        for count in [0,1,2,3,8,width]:
            for _ in range(30):
                x=np.zeros(width);indices=rng.choice(width,count,replace=False);x[indices]=rng.choice([-.5,.5,2.],count)
                assert ballot_classify(x)==(sorted(indices.tolist()) if count<=2 else None)
        for index in range(width):
            x=np.zeros(width);x[index]=1
            assert ballot_classify(x)==[index]


def test_range_rejection_and_signed_zero():
    for value in [2.**-60,2.**60,np.inf,np.nan]:
        x=np.zeros(512);x[-1]=value
        assert ballot_classify(x) is None
    assert ballot_classify(np.full(128,-0.))==[]
