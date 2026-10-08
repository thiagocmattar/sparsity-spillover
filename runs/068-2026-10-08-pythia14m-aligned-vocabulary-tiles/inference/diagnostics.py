"""Native head statistics and independent tensor-core work counts, pooled as integers."""
import torch


def counts(value, threshold, m, n, vocabulary=50304):
    if m != 16: raise ValueError('Aligned M16 consumer required')
    x=value.reshape(-1,128)
    live=x[::16,::16]>threshold
    certificate=live[:,None,:,None].expand(-1,16,-1,16).reshape_as(x)
    if bool(x.masked_select(~certificate).ne(0).any()):
        raise ValueError('FP32 corner certificate does not imply an empty 16x16 tile')
    union=live
    columns=(vocabulary+n-1)//n
    mma_per_k=columns*(n//8)
    potential=(x.shape[0]//m)*8*mma_per_k
    issued=int(union.sum())*mma_per_k
    zero=int(x.eq(0).sum())
    return {'elements':x.numel(),'exact_zero':zero,
            'near_001':int(x.abs().le(.001).sum()),'near_01':int(x.abs().le(.01).sum()),
            'sum_squares':float(x.double().square().sum()),'sum_abs':float(x.double().abs().sum()),
            'gate_tiles':live.numel(),'empty_gate_tiles':int((~live).sum()),
            'consumer_tiles':union.numel(),'empty_consumer_tiles':int((~union).sum()),
            'potential_mma':potential,'issued_mma':issued,'skipped_mma':potential-issued,
            'logical_head_products':x.numel()*vocabulary,'activation_zero_products':zero*vocabulary,
            'padded_scalar_products':potential*16*8*16}


def pool(rows):
    return {name:sum(row[name] for row in rows) for name in rows[0]}
