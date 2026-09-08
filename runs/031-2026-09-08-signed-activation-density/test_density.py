"""Scientific checks: signs, exact zeros, bin boundaries, pooling and hooks."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from density import GROUPS, SITES, SignedHistogram, edges, nonzero_density, pool, read_json, validate_row, write_json
from sparsity_research.metrics import ActivationAccumulator


def collect(values):
    moments = ActivationAccumulator()
    histogram = SignedHistogram(torch=torch,device='cpu')
    moments.update(values,torch=torch)
    histogram.update(values)
    return histogram.rows(moments.rows())


@pytest.mark.parametrize('dtype',[torch.float16,torch.float32])
def test_signed_bins_keep_boundary_values_tails_and_separate_zero(dtype):
    values = torch.tensor([-9,-8,-.5,-.05,-.001,-0.,0.,.001,.05,.5,8,9],dtype=dtype)
    row = collect({'h.layer_0':values})[0]
    observed = values.float().numpy()
    inside = observed[(observed!=0)&(observed>=-8)&(observed<=8)]
    expected,_ = np.histogram(inside,bins=edges())
    assert row['histogram']==expected.tolist()
    assert row['exact_zero_count']==2 and row['underflow']==row['overflow']==1
    assert np.sum(nonzero_density(row)*np.diff(edges().astype(np.float64)))==pytest.approx(8/12)


def test_streaming_and_element_pooling_preserve_counts():
    values = {'h.layer_0':torch.tensor([0.,0.,0.,1.]),'m.layer_0':torch.tensor([2.])}
    combined = pool(collect(values),'FFN activations',('h','m'))
    assert combined['total']==5 and combined['exact_zero_count']==3
    assert combined['site_element_weights']=={'h':.8,'m':.2}
    assert np.sum(nonzero_density(combined)*np.diff(edges().astype(np.float64)))==pytest.approx(.4)
    moments = ActivationAccumulator()
    histogram = SignedHistogram(torch=torch,device='cpu')
    for _ in range(2):
        moments.update(values,torch=torch)
        histogram.update(values)
    doubled = pool(histogram.rows(moments.rows()),'FFN activations',('h','m'))
    assert doubled['histogram']==[2*n for n in combined['histogram']]
    assert doubled['total']==10 and doubled['exact_zero_count']==6
    with pytest.raises(ValueError,match='Missing sites'):
        pool(collect(values),'Attention activations',GROUPS['Attention activations'])


def test_nonfinite_and_corrupt_counts_are_rejected():
    with pytest.raises(ValueError,match='Nonfinite'):
        collect({'h.layer_0':torch.tensor([0.,1.,float('nan')])})
    row = collect({'h.layer_0':torch.tensor([0.,1.])})[0]
    bad = copy.deepcopy(row)
    bad['histogram'][0] += 1
    with pytest.raises(ValueError,match='partition'):
        validate_row(bad)


def test_signed_artifact_compressed_roundtrip(tmp_path):
    row = collect({'h.layer_0':torch.tensor([-.5,0.,.05])})[0]
    path = tmp_path/'counts.json.gz'
    write_json(path,row)
    assert read_json(path)==row
    original = path.read_bytes()
    write_json(path,row)
    assert path.read_bytes()==original


def test_real_neox_histograms_capture_post_gate_signed_ports():
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    from sparsity_research.capture import ActivationCapture
    from sparsity_research.pythia import apply_activation_topology
    config = GPTNeoXConfig(vocab_size=32,hidden_size=8,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=2,max_position_embeddings=16,
        rotary_pct=.25,hidden_dropout=0.,attention_dropout=0.)
    config.topology_id = 'A7-Z-POST'
    config.site_gates = {site:{'operator':'one_sided_threshold' if site in ('a','m','h','z')
                              else 'symmetric_threshold','kappa':.05}
                         for site in ('a','m','h','z','q_post','k_post','v')}
    model = apply_activation_topology(GPTNeoXForCausalLM(config),torch=torch).eval()
    with torch.no_grad(), ActivationCapture(model,list(SITES),torch=torch) as capture:
        output = model(input_ids=torch.tensor([[1,2,3,4]]),labels=torch.tensor([[1,2,3,4]]))
        assert torch.isfinite(output.loss)
        assert set(capture.activations)=={s+'.layer_0' for s in SITES}
        for site in SITES:
            x = capture.activations[site+'.layer_0']
            nonzero = x[x!=0]
            assert torch.all(nonzero>=.05) if site in ('h','m') else torch.all(nonzero.abs()>=.05)
        rows = collect(capture.activations)
    ffn = pool(rows,'FFN activations',GROUPS['FFN activations'])
    attention = pool(rows,'Attention activations',GROUPS['Attention activations'])
    assert ffn['site_element_weights']=={'h':.8,'m':.2}
    assert list(attention['site_element_weights'].values())==[1/3]*3
