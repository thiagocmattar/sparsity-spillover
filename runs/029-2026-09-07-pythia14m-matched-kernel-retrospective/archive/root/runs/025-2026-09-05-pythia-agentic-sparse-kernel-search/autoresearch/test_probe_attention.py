"""CPU dispatch/mask/branch-record checks; CUDA kernels remain unqualified."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

spec = importlib.util.spec_from_file_location("run025_attention_probe_tested", Path(__file__).with_name("probe_attention.py"))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_mode_switch_preserves_attention_and_linear_identity():
    calls = []
    model = SimpleNamespace(set_attn_implementation=lambda name: calls.append(("attention", name)))
    linears = SimpleNamespace(set_mode=lambda mode: calls.append(("linears", mode)))
    adapter = probe.AttentionAdapter(model, linears)
    adapter.set_mode("candidate")
    adapter.set_mode("native")
    assert calls == [("attention", probe.ATTENTION_NAME), ("linears", "candidate"),
                     ("attention", "sdpa"), ("linears", "native")]
    with pytest.raises(ValueError):
        adapter.set_mode("fake")


def test_branch_positions_counts_and_stale_flag_rejection():
    workspace = SimpleNamespace(fast_tiles=torch.tensor([[[0, 1], [1, 0]]], dtype=torch.uint8),
                                block_m=16, shape=(1, 2, 32, 32), bytes=8192)
    model = SimpleNamespace(gpt_neox=SimpleNamespace(layers=[
        SimpleNamespace(attention=SimpleNamespace(_run025_k002_workspace=workspace))]))
    row = probe.branch_snapshot(model)[0]
    assert row["fast_tile_flat_indices"] == [1, 2]
    assert row["fast_tiles"] == 2 and row["total_tiles"] == 4
    assert row["shape_b_h_query_tiles"] == [1, 2, 2]
    probe.invalidate_branch_flags(model)
    with pytest.raises(RuntimeError, match="unwritten"):
        probe.branch_snapshot(model)
    del model.gpt_neox.layers[0].attention._run025_k002_workspace
    with pytest.raises(RuntimeError, match="did not execute"):
        probe.branch_snapshot(model)


@pytest.mark.parametrize("padded", [False, True])
def test_actual_a7_model_mask_registry_matches_native_cpu_fallback(padded):
    from sparsity_research.pythia import apply_activation_topology
    from transformers.models.gpt_neox.modeling_gpt_neox import ALL_ATTENTION_FUNCTIONS
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    cfg = GPTNeoXConfig(vocab_size=32, hidden_size=64, intermediate_size=128,
                       num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=16,
                       rotary_pct=.5, hidden_dropout=0., attention_dropout=0.)
    cfg.topology_id = "A7-Z-POST"
    cfg.site_gate = None
    cfg.site_gates = {site: {"operator": "symmetric_threshold" if site in ("q_post", "k_post", "v")
                            else "one_sided_threshold", "kappa": .05}
                      for site in ("a", "m", "h", "z", "q_post", "k_post", "v")}
    torch.manual_seed(2503)
    model = apply_activation_topology(GPTNeoXForCausalLM(cfg), torch=torch).bfloat16().eval()
    attention = probe.load_attention()
    probe.register_attention(attention.interface)
    assert probe.ATTENTION_NAME in ALL_ATTENTION_FUNCTIONS
    assert ALL_MASK_ATTENTION_FUNCTIONS[probe.ATTENTION_NAME] is ALL_MASK_ATTENTION_FUNCTIONS["sdpa"]
    adapter = probe.AttentionAdapter(model)
    ids = torch.tensor([[1, 2, 3, 4]])
    kwargs = {"attention_mask": torch.tensor([[1, 1, 0, 1]])} if padded else {}
    with torch.inference_mode():
        adapter.set_mode("native")
        expected = model(input_ids=ids, use_cache=False, **kwargs).logits.clone()
        adapter.set_mode("candidate")
        actual = model(input_ids=ids, use_cache=False, **kwargs).logits
    assert torch.equal(expected, actual)
    # CPU uses the explicit dense fallback, and must NOT pass the CUDA branch proof.
    with pytest.raises(RuntimeError, match="did not execute"):
        probe.branch_snapshot(model)
