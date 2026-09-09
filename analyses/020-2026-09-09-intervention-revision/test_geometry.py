import numpy as np
import pytest
import torch
from geometry import ideal_pressure, reach
from sparsity_research.pressure import apply_ol1_correction
from sparsity_research.ceilings import architecture_ceiling
from sparsity_research.sites import FixedOneSidedThreshold, FixedSymmetricThreshold


@pytest.mark.parametrize("w", [[2,0], [0,2], [-1,2], [-2,0], [0,0]])
def test_protection_cap_and_alignment_optimum(w):
    u = np.array([1.,0.]); w = np.array(w, dtype=float)
    d = ideal_pressure(u, w, weight=100)
    assert u @ d >= -1e-14
    assert np.linalg.norm(d) <= 1+1e-14
    if u @ w < 0:
        assert abs(u @ d) < 1e-14
    # Independent feasible circle sample checks the maximum-alignment claim.
    angles = np.linspace(-np.pi/2, np.pi/2, 10001)
    feasible = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    assert np.max(feasible @ w) <= w @ d + 1e-12


def test_saturation_boundary_and_positive_scaling():
    u, w = np.array([1.,0.]), np.array([-2.,4.])
    np.testing.assert_allclose(ideal_pressure(u,w,.125), [0,.5])
    boundary = ideal_pressure(u,w,.25)
    np.testing.assert_allclose(boundary, [0,1])
    for weight in [.25,1,100]:
        np.testing.assert_allclose(ideal_pressure(u,w,weight), boundary)
    np.testing.assert_allclose(ideal_pressure(u,w,1), ideal_pressure(u,7/4*w,1))
    assert not np.allclose(ideal_pressure(u,w,.01), ideal_pressure(u,7/4*w,.01))
    np.testing.assert_array_equal(ideal_pressure([0,0], w), [0,0])


def test_protection_does_not_preserve_true_task_gradient_or_combined_norm():
    u=np.array([1.,1.]); g=np.array([1.,0.])
    d=ideal_pressure(u,[-2,0],10)
    np.testing.assert_allclose(d,[-1,1])
    assert u @ d == 0 and g @ (u+d) == 0 and g @ u == 1
    assert np.linalg.norm(u+d) > np.linalg.norm(u)


def executed_correction(u,w,eps=1e-12):
    p=torch.nn.Parameter(torch.zeros(len(u),dtype=torch.float32))
    opt=torch.optim.AdamW([p],lr=1.,betas=(0.,0.),eps=0.,weight_decay=0.)
    opt.state[p].update(step=torch.tensor(1.), exp_avg=torch.tensor(u,dtype=torch.float32),
                        exp_avg_sq=torch.ones_like(p))
    metrics=apply_ol1_correction(opt,[p],[torch.tensor(u)],[torch.tensor(w)],
                               pressure_weight=100.,step_budget=1.,eps=eps)
    return -p.detach().numpy(), metrics


def test_stabilized_rule_matches_large_vectors_but_small_norm_guard_is_real():
    actual, metrics=executed_correction([1,1],[-2,0])
    np.testing.assert_allclose(actual,ideal_pressure([1,1],[-2,0],100),atol=1e-6)
    assert metrics['projection_applied']
    actual, metrics=executed_correction([1e-7,0],[-1.,0])
    assert not metrics['projection_applied']
    assert actual[0] < 0  # q <= epsilon leaves this conflict unprojected.


@pytest.mark.parametrize("layers,width", [(6,128),(6,512),(24,1024)])
def test_reach_against_integer_operation_inventory_and_union(layers,width):
    expected=reach(layers,width,2048,50304)
    for topology,key in [('A1-H','h'),('A2','mh'),('A4-Z','a4'),('A7-Z-POST','a7')]:
        actual=architecture_ceiling(topology,layers=layers,hidden_size=width,ffn_size=4*width,
                                    sequence_length=2048,vocabulary_size=50304)
        assert actual['model_product_count']==expected['model_products']
        assert actual['R_model_max_fraction']==pytest.approx(expected[key])
    a6=architecture_ceiling('A6-POST',layers=layers,hidden_size=width,ffn_size=4*width,
                            sequence_length=2048,vocabulary_size=50304)
    assert 'attention_output_projection' in a6['reachable_operations']  # V=0 => PV=0
    assert a6['R_model_max_fraction']==pytest.approx(expected['a7'])  # z adds no duplicate credit.
    assert expected['a7']-expected['a4']==pytest.approx(expected['attention_gap'])


def test_width_depth_sequence_and_vocabulary_trends():
    base=reach(6,128,2048,50304)
    assert reach(6,512,2048,50304)['attention_gap'] < base['attention_gap']
    assert reach(6,128,4096,50304)['attention_gap'] > base['attention_gap']
    assert reach(12,128,2048,50304)['a7'] > base['a7']
    assert reach(6,128,2048,100608)['a7'] < base['a7']


@pytest.mark.parametrize("symmetric", [False,True])
def test_exact_threshold_values_and_detached_boundary_derivatives(symmetric):
    x=torch.tensor([-.5001,-.5,-.4999,0,.4999,.5,.5001],requires_grad=True)
    module=(FixedSymmetricThreshold if symmetric else FixedOneSidedThreshold)(.5)
    y=module(x); y.sum().backward()
    mask=x.detach().abs()>=.5 if symmetric else x.detach()>=.5
    torch.testing.assert_close(y,x.detach()*mask)
    torch.testing.assert_close(x.grad,mask.float())
    z=torch.tensor([0.],requires_grad=True)
    (FixedSymmetricThreshold if symmetric else FixedOneSidedThreshold)(0)(z).sum().backward()
    assert z.grad.item()==1
    z.grad=None; torch.relu(z).sum().backward(); assert z.grad.item()==0
