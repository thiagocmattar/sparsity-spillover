import sys
from pathlib import Path
import pytest
import torch
import transformers

sys.path.insert(0, str(Path(__file__).parent))
from common import module, HERE
from sparsity_research.pythia import apply_activation_topology
from progress import rows

candidate = module('test_k017', HERE/'candidates/k017/candidate.py')


def reference_backend(x, wt, bias, out, kappa, elements, warps):
    gated = x.masked_fill(x < kappa, 0)
    out.copy_((gated.float() @ wt.float() + (bias.float() if bias.numel() else 0)).bfloat16())


def test_threshold_equality_bias_signed_and_workspace_reuse():
    layer = torch.nn.Linear(33, 129).bfloat16().eval()
    fused = candidate.GatedLinear(layer, .5, backend=reference_backend).eval()
    with torch.inference_mode():
        for scale in [1, 0, 2]:
            x = (torch.randn(2, 3, 33)*scale).bfloat16()
            x[..., 0] = .5
            actual = fused(x).clone()
            expected = (x.masked_fill(x < .5, 0).float() @ layer.weight.float().T + layer.bias.float()).bfloat16()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('sites', [('h',), ('z',), ('a', 'm', 'h', 'z')])
def test_complete_topology_fusion_and_restore(sites):
    cfg = transformers.GPTNeoXConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
             num_attention_heads=4, vocab_size=97, max_position_embeddings=64,
             attention_dropout=0., hidden_dropout=0.)
    cfg.topology_id = 'A7-Z-POST'
    cfg.site_gates = {s: {'operator': 'one_sided_threshold' if s in {'a','m','h','z'} else 'symmetric_threshold',
                         'kappa': .5} for s in ['a','m','h','q_post','k_post','v','z']}
    model = apply_activation_topology(transformers.GPTNeoXForCausalLM(cfg), torch=torch).bfloat16().eval()
    adapter = candidate.Adapter(model, sites, backend=reference_backend)
    with torch.inference_mode():
        for _ in range(2):
            ids = torch.randint(0, 97, (1, 16))
            adapter.set_mode(False)
            expected = model(ids).logits
            adapter.set_mode(True)
            actual = model(ids).logits
            torch.testing.assert_close(actual, expected, atol=.02, rtol=.02)
            adapter.set_mode(False)
            torch.testing.assert_close(model(ids).logits, expected, rtol=0, atol=0)


def test_failed_trial_stays_visible_without_improving_best():
    trials = [dict(attempt='001', idea='valid', status='complete', qualified=True, primary_speedup=1.2),
              dict(attempt='002', idea='wrong', status='complete', qualified=False, primary_speedup=8.),
              dict(attempt='003', idea='build failed', status='failed', qualified=False)]
    progress = list(rows(trials))
    assert len(progress) == 3
    assert [r['best_so_far_matched_dense'] for r in progress] == [1.2]*3
