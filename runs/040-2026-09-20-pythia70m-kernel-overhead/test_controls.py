"""Gate preservation, native branch rounding, layout and declared coverage."""
import importlib.util
import json
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from controls import NativeJoint, NativeRope, original_norms, substitute, MODES
from sparsity_research.sites import FixedOneSidedThreshold, FixedSymmetricThreshold


@pytest.mark.parametrize('gated', [False, True])
def test_joint_preserves_threshold_equality_and_bf16_addition(gated):
    w2, wo = [torch.nn.Linear(4, 4, dtype=torch.bfloat16) for _ in range(2)]
    op = NativeJoint(SimpleNamespace(w2=w2, wo=wo, gh=gated, gz=gated, th=.5, tz=.5))
    h = torch.tensor([[.25, .5, .75, -1]], dtype=torch.bfloat16)
    z = h.flip(-1)
    residual = torch.tensor([[1, -2, .00390625, 100]], dtype=torch.bfloat16)
    hh, zz = (torch.where(x >= .5, x, 0) if gated else x for x in (h, z))
    expected = (w2(hh) + wo(zz)) + residual
    assert torch.equal(op(h, z, residual), expected)


def test_native_norm_restores_gates_and_preserves_hooks():
    first, second = torch.nn.LayerNorm(4), torch.nn.LayerNorm(4)
    ga, gm = FixedOneSidedThreshold(.5), FixedOneSidedThreshold(.5)
    first.register_forward_hook(lambda m, a, y: ga(y))
    second.register_forward_hook(lambda m, a, y: gm(y))
    layer = SimpleNamespace(input_layernorm=first, post_attention_layernorm=second, a_gate=ga, m_gate=gm)
    model = SimpleNamespace(gpt_neox=SimpleNamespace(layers=[layer]))
    x = torch.tensor([[-2., -1., .5, 1.]])
    expected = first(x).clone(), second(x).clone()
    saved = original_norms(model)
    for norm in (first, second): norm.forward = MethodType(lambda self, value: value + 99, norm)
    for gate in (ga, gm): gate.forward = MethodType(lambda self, value: value, gate)
    substitute(model, 'native-norm', saved)
    assert torch.equal(first(x), expected[0]) and torch.equal(second(x), expected[1])
    assert (first(x) == 0).any()


def test_native_rope_gates_rotated_values_and_keeps_qkv_layout():
    attn = SimpleNamespace(head_size=4, q_post_gate=FixedSymmetricThreshold(.5),
                           k_post_gate=FixedSymmetricThreshold(.5), v_gate=FixedSymmetricThreshold(.5))
    raw = torch.arange(24, dtype=torch.float32).reshape(1, 1, 24) / 10 - 1
    cos, sin = torch.zeros(1, 1, 2), torch.ones(1, 1, 2)
    q, k, v = NativeRope(attn)(raw, cos, sin)
    ref_q, ref_k, ref_v = raw.view(1, 1, 2, 12).transpose(1, 2).chunk(3, dim=-1)
    def rotate_and_gate(x):
        rotated = torch.cat((-x[..., 1:2], x[..., :1], x[..., 2:]), dim=-1)
        return torch.where(rotated.abs() >= .5, rotated, 0)
    assert torch.equal(q, rotate_and_gate(ref_q))
    assert torch.equal(k, rotate_and_gate(ref_k))
    assert torch.equal(v, torch.where(ref_v.abs() >= .5, ref_v, 0))


def test_complete_reproducible_diagnostic_matrix():
    spec = importlib.util.spec_from_file_location('run040_execute', HERE / '03_execute.py')
    execute = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(execute)
    assert len(execute.jobs()) == 54 and len(execute.jobs(True)) == 18
    assert set(execute.jobs()) == {(c, m, r) for c in ('c00', 'c21') for m in MODES for r in (1, 2, 3)}
    assert execute.jobs() == execute.jobs()


def test_retained_cohort_and_complete_validation():
    cfg = json.loads((HERE / 'config.json').read_text())
    manifest = json.loads((HERE / 'provenance/inputs.json').read_text())
    assert [(r['id'], r['family']) for r in manifest['checkpoints']] == [('c00', 'A0'), ('c21', 'A7+OL1@h')]
    assert cfg['diagnostic_modes'] == list(MODES)
    assert (cfg['validation_blocks'], cfg['validation_documents'], cfg['excluded_tail_tokens']) == (338, 500, 1444)
    assert (cfg['timing_inputs'], cfg['timing_passes'], cfg['process_replicates']) == (64, 7, 3)
