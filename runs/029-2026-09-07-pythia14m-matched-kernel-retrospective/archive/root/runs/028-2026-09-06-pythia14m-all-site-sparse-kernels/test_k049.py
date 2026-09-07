"""Support-mask traversal equals ascending nonempty K16 fragments."""
import numpy as np


def row_mask(row):
    bits=0
    for base in range(0,len(row),32):
        ballot=sum(1<<lane for lane,value in enumerate(row[base:base+32]) if value!=0)
        if ballot&0xffff:bits|=1<<(base//16)
        if ballot&0xffff0000:bits|=1<<(base//16+1)
    return bits


def test_all_feature_positions_and_mask_order():
    for width in [128,512]:
        for index in range(width):
            row=np.zeros(width);row[index]=1
            assert row_mask(row)==1<<(index//16)
    rng=np.random.default_rng(2849)
    for _ in range(100):
        values=rng.choice([0.,0.,0.,1.,-1.],(8,512))
        simple=np.count_nonzero(values,axis=1)<=2
        mask=0
        for row in values[~simple]:mask|=row_mask(row)
        observed=[]
        while mask:
            observed.append((mask&-mask).bit_length()-1);mask&=mask-1
        expected=np.flatnonzero(np.any(values[~simple].reshape(-1,32,16)!=0,axis=(0,2))).tolist()
        assert observed==expected


def test_simple_rows_do_not_schedule_fallback_tiles():
    values=np.zeros((8,512));values[:,511]=1;values[0,[0,16]]=2
    mask=0
    for row in values[np.count_nonzero(values,axis=1)>2]:mask|=row_mask(row)
    assert mask==(1<<0)|(1<<1)|(1<<31)
