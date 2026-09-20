"""CPU tests for rotation, gate enforcement, and exact A7 SDPA dispatch."""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

SPEC = importlib.util.spec_from_file_location("run025_dense_probe", Path(__file__).with_name("probe.py"))
PROBE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PROBE
SPEC.loader.exec_module(PROBE)


def test_rotating_stage_and_full_logits_are_paired():
    buffer = torch.zeros((1, 4), dtype=torch.long)
    inputs = [torch.full_like(buffer, index) for index in range(3)]
    calls = []
    def forward(ids):
        calls.append(ids.clone())
        return F.one_hot(ids, num_classes=8).float()
    runners = {mode: PROBE.DenseRunner(forward, buffer, "native", cuda=False)
               for mode in ("native", "sibling")}
    samples = PROBE.paired_probe(runners, inputs, passes=2, seed=2504, cuda=False)
    assert len(samples) == 12
    for sample, actual_ids in zip(samples, calls, strict=True):
        assert torch.equal(actual_ids, inputs[sample["input_index"]])
        assert sample["output_shape"] == [1, 4, 8]
        assert sample["staging_ms_excluded"] >= 0
    assert all(len([row for row in samples if row["input_index"] == index and row["mode"] == mode]) == 2
               for index in range(3) for mode in runners)


def test_cpu_graph_is_rejected_and_unknown_mode_fails():
    buffer = torch.zeros((1, 4))
    with pytest.raises(ValueError, match="CUDA"):
        PROBE.DenseRunner(lambda x: x, buffer, "graph", cuda=False).prepare()
    with pytest.raises(ValueError, match="Unknown"):
        PROBE.DenseRunner(lambda x: x, buffer, "unlisted", cuda=False).prepare()


def test_quality_gate_detects_wrong_outputs_and_pools_shifted_tokens():
    buffer = torch.zeros((1, 4), dtype=torch.long)
    def forward(ids):
        return F.one_hot(ids, num_classes=8).float()
    runners = {"native": PROBE.DenseRunner(forward, buffer, "native", cuda=False),
               "correct": PROBE.DenseRunner(forward, buffer, "native", cuda=False),
               "wrong": PROBE.DenseRunner(lambda ids: forward(ids) + 2, buffer, "native", cuda=False)}
    inputs = [torch.tensor([[0, 1, 2, 3]]), torch.tensor([[4, 5, 6, 7]])]
    result = PROBE.compare_inputs(runners, inputs, PROBE.config()["calibration"])
    assert result["prediction_tokens"] == 6
    assert result["pass"] == {"native": True, "correct": True, "wrong": False}
    assert result["loss_delta"]["correct"] == 0
    # Uniform offset preserves CE but must still fail the logit gate.
    assert abs(result["loss_delta"]["wrong"]) < 1e-6


def test_a7_custom_forward_uses_sdpa_with_post_rope_gated_operands(monkeypatch):
    from sparsity_research.pythia import apply_activation_topology, topology_metadata
    config = GPTNeoXConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=16,
        rotary_pct=0.5, hidden_dropout=0., attention_dropout=0.)
    config.topology_id = "A7-Z-POST"
    config.site_gate = None
    config.site_gates = {site: {"operator": "symmetric_threshold" if site in {"q_post", "k_post", "v"}
                              else "one_sided_threshold", "kappa": .05}
                         for site in ("a", "m", "h", "q_post", "k_post", "v", "z")}
    torch.manual_seed(2503)
    model = apply_activation_topology(GPTNeoXForCausalLM(config), torch=torch).eval()
    model.set_attn_implementation("sdpa")
    attn = model.gpt_neox.layers[0].attention
    captured, sdpa_calls, handles = {}, [], []
    for name in ("q_post", "k_post", "v"):
        def save(_module, _inputs, output, site=name):
            captured[site] = output.detach().clone()
        handles.append(getattr(attn, f"{name}_site").register_forward_hook(save))
    original = F.scaled_dot_product_attention
    def spy(query, key, value, **kwargs):
        sdpa_calls.append((query.clone(), key.clone(), value.clone(), kwargs))
        return original(query, key, value, **kwargs)
    monkeypatch.setattr(F, "scaled_dot_product_attention", spy)
    try:
        with torch.inference_mode():
            result = model(input_ids=torch.tensor([[1, 2, 3, 4]]), use_cache=False, output_attentions=False)
    finally:
        for handle in handles:
            handle.remove()
    assert torch.isfinite(result.logits).all()
    assert topology_metadata(model)["qk_placement"] == "post_rope"
    assert len(sdpa_calls) == 1
    for site, tensor in zip(("q_post", "k_post", "v"), sdpa_calls[0][:3], strict=True):
        assert torch.equal(tensor, captured[site])
        assert torch.all(tensor[tensor != 0].abs() >= .05)
    assert sdpa_calls[0][3]["is_causal"] is True
    assert sdpa_calls[0][3]["attn_mask"] is None


def test_zero_query_sdpa_preserves_causal_prefix_average():
    query = torch.zeros(1, 1, 4, 4)
    key = torch.arange(16).float().reshape(1, 1, 4, 4)
    value = torch.arange(16).float().reshape(1, 1, 4, 4)
    actual = F.scaled_dot_product_attention(query, key, value, is_causal=True)
    expected = torch.stack([value[:, :, :i + 1].mean(2) for i in range(4)], dim=2)
    assert torch.allclose(actual, expected)
