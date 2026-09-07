"""Independent mathematical shortcut tests; no claim of CUDA equivalence."""
import torch


def test_zero_query_is_causal_mean_not_zero():
    v=torch.tensor([[1.,-2.],[3.,0.],[0.,5.]],dtype=torch.float64)
    q=torch.zeros_like(v)
    k=torch.randn_like(v)
    scores=q@k.T
    scores.masked_fill_(torch.ones(3,3,dtype=torch.bool).triu(1),float('-inf'))
    expected=scores.softmax(-1)@v
    prefix=v.cumsum(0)/torch.arange(1,4,dtype=torch.float64)[:,None]
    torch.testing.assert_close(expected,prefix)
    assert not torch.equal(expected,torch.zeros_like(expected))


def test_zero_values_keep_normalization_mass():
    scores=torch.tensor([[0.,0.,0.]],dtype=torch.float64)
    v=torch.tensor([[3.],[0.],[0.]],dtype=torch.float64)
    assert (scores.softmax(-1)@v).item()==1.
    assert (scores[:,:1].softmax(-1)@v[:1]).item()==3.


def test_zero_operand_union_does_not_drop_nonzeros():
    q=torch.tensor([0.,2.,-1.,0.],dtype=torch.float64)
    k=torch.tensor([[4.,0.,3.,0.],[0.,4.,-2.,1.]],dtype=torch.float64)
    dense=q@k.T
    explicit=torch.tensor([sum(float(q[d]*row[d]) for d in range(4) if q[d]!=0 and row[d]!=0) for row in k],dtype=torch.float64)
    torch.testing.assert_close(dense,explicit)
def test_native_mma_counter_unit_and_causal_padding():
    # The ordinary D32 Flash path uses 128x128 tiles: 4 warps x2 row atoms x16 col atoms
    # x2 reduction atoms; each MMA represents 16x8x16 scalar FMAs.
    blocks=2048//128
    mma_count=4*sum(range(1,blocks+1))*4*2*16*2
    assert mma_count*2048==285212672
    assert mma_count*2048>4*32*2048*2049//2  # padded tiles != R_model denominator


def test_splitkv_padding_and_partition_count():
    # Native B1 H4 T2048 D32 uses 64x256 tiles and 2 KV splits.
    visits=sum((64*(m+1)+255)//256 for m in range(32))
    split_visits=sum(max(0,min((64*(m+1)+255)//256,(s+1)*4)-s*4) for m in range(32) for s in range(2))
    assert visits==split_visits==144
    qk_mmas=4*visits*4*1*32*2
    pv_mmas=4*visits*4*1*4*16
    assert qk_mmas==pv_mmas==147456
    assert qk_mmas*2048==301989888


def test_safe_bf16_prefix_grid_is_exact_in_fp32():
    # Bound applies to <=1024 terms. All BF16 values in this range are
    # multiples of 1/256, and every partial sum has magnitude <=2^24 grid units.
    assert 64*1024*256==2**24
    gen=torch.Generator().manual_seed(2901)
    x=(torch.randint(-16384,16385,(1024,),generator=gen).float()/256).bfloat16()
    x=torch.where(x.abs()>=.5,x,0).float()
    assert torch.equal(x*256,(x*256).round())
    exact=x.double().cumsum(0)
    assert torch.equal(x.cumsum(0).double(),exact)
    assert x.sum().double()==exact[-1]
    assert x.flip(0).sum().double()==exact[-1]


def test_k034_tile_output_coverage_and_shared_swizzle():
    outputs=[]
    for tid in range(128):
        lane,warp=tid%32,tid//32
        for nn in range(4):
            for e in range(4):
                outputs.append(((warp//2)*16+lane//4+(e//2)*8,
                                (warp%2)*32+nn*8+(lane%4)*2+e%2))
    assert len(outputs)==len(set(outputs))==32*64
    assert set(outputs)=={(r,c) for r in range(32) for c in range(64)}
    for row in range(64):
        assert {row*32+(word^((row&7)*4)) for word in range(32)}==set(range(row*32,(row+1)*32))


def test_segmented_safe_grid_prefix_matches_full_sum():
    gen=torch.Generator().manual_seed(3501)
    values=(torch.randint(-128,129,(1024,32),generator=gen).float()*.5).bfloat16().float()
    local=values.reshape(16,64,32).cumsum(1)
    reconstructed=[]
    base=torch.zeros(32)
    for segment in range(16):
        reconstructed.append(local[segment]+base)
        base=base+local[segment,-1]
    assert torch.equal(torch.cat(reconstructed).double(),values.double().cumsum(0))
