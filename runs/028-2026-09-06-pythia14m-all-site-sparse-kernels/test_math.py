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
