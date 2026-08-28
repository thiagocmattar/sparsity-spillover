import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

from sparsity_research.capture import ActivationCapture
from sparsity_research.pythia import apply_activation_topology, topology_metadata


def test_real_gpt_neox_forward_exposes_and_gates_exact_a6_post_sites():
    config = GPTNeoXConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=16,
        rotary_pct=0.25,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    config.topology_id = "A6-POST"
    config.site_gate = {"operator": "symmetric_threshold", "kappa": 0.1}
    model = apply_activation_topology(GPTNeoXForCausalLM(config), torch=torch)
    sites = ["a", "m", "h", "q_post", "k_post", "v"]
    with ActivationCapture(model, sites, torch=torch) as capture:
        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        output = model(input_ids=input_ids, labels=input_ids)

    assert torch.isfinite(output.loss)
    assert set(capture.activations) == {f"{site}.layer_0" for site in sites}
    assert topology_metadata(model) == {
        "topology_id": "A6-POST",
        "active_sites": sites,
        "qk_placement": "post_rope",
        "site_gate": {"operator": "symmetric_threshold", "kappa": 0.1},
    }
