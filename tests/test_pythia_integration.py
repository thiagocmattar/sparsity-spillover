import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

from sparsity_research.capture import ActivationCapture
from sparsity_research.pressure import activation_l1
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


def test_z_is_captured_after_head_concatenation_immediately_before_wo():
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
    config.topology_id = "A4-Z"
    config.site_gate = {"operator": "one_sided_threshold", "kappa": 0.05}
    model = apply_activation_topology(GPTNeoXForCausalLM(config), torch=torch)
    dense_inputs = []
    handle = model.gpt_neox.layers[0].attention.dense.register_forward_pre_hook(
        lambda _module, inputs: dense_inputs.append(inputs[0].detach().clone())
    )
    try:
        with ActivationCapture(model, ["z"], torch=torch) as capture:
            input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
            output = model(input_ids=input_ids, labels=input_ids)
    finally:
        handle.remove()

    captured = capture.activations["z.layer_0"]
    assert torch.isfinite(output.loss)
    assert captured.shape == (1, 4, 8)
    assert len(dense_inputs) == 1
    assert torch.equal(captured, dense_inputs[0])
    assert torch.all(captured[captured != 0] >= 0.05)


def test_a4z_pressure_capture_observes_all_post_threshold_site_layer_tensors():
    config = GPTNeoXConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=16,
        rotary_pct=0.25,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    config.topology_id = "A4-Z"
    config.site_gate = {"operator": "one_sided_threshold", "kappa": 0.05}
    model = apply_activation_topology(GPTNeoXForCausalLM(config), torch=torch)
    sites = ["a", "m", "h", "z"]
    with ActivationCapture(model, sites, torch=torch) as capture:
        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        output = model(input_ids=input_ids, labels=input_ids)
        pressure = activation_l1(capture.activations)

    assert torch.isfinite(output.loss)
    assert torch.isfinite(pressure)
    assert set(capture.activations) == {
        f"{site}.layer_{layer}" for site in sites for layer in range(2)
    }
    for value in capture.activations.values():
        assert torch.all(value[value != 0] >= 0.05)
    expected = torch.stack(
        [value.float().abs().mean() for value in capture.activations.values()]
    ).mean()
    assert torch.equal(pressure, expected)


def test_real_gpt_neox_mixed_a7_z_post_gates_use_exact_post_rope_operands():
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
    branch = {"operator": "one_sided_threshold", "kappa": 0.1}
    attention = {"operator": "symmetric_threshold", "kappa": 0.1}
    site_gates = {
        "a": branch,
        "m": branch,
        "h": branch,
        "q_post": attention,
        "k_post": attention,
        "v": attention,
        "z": branch,
    }
    config.topology_id = "A7-Z-POST"
    config.site_gate = None
    config.site_gates = site_gates
    model = apply_activation_topology(GPTNeoXForCausalLM(config), torch=torch)
    dense_inputs = []
    handle = model.gpt_neox.layers[0].attention.dense.register_forward_pre_hook(
        lambda _module, inputs: dense_inputs.append(inputs[0].detach().clone())
    )
    try:
        with ActivationCapture(model, list(site_gates), torch=torch) as capture:
            input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
            output = model(input_ids=input_ids, labels=input_ids)
    finally:
        handle.remove()

    assert torch.isfinite(output.loss)
    assert topology_metadata(model)["site_gates"] == site_gates
    assert torch.equal(capture.activations["z.layer_0"], dense_inputs[0])
    for site in ("a", "m", "h", "z"):
        values = capture.activations[f"{site}.layer_0"]
        assert torch.all(values[values != 0] >= 0.1)
    for site in ("q_post", "k_post", "v"):
        values = capture.activations[f"{site}.layer_0"]
        assert torch.all(values[values != 0].abs() >= 0.1)


def test_mixed_kappa_zero_matches_a4z_values_and_parameter_gradients_exactly():
    def make_config():
        return GPTNeoXConfig(
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

    baseline_config = make_config()
    baseline_config.topology_id = "A4-Z"
    baseline_config.site_gate = {"operator": "one_sided_threshold", "kappa": 0.0}
    baseline = apply_activation_topology(GPTNeoXForCausalLM(baseline_config), torch=torch)

    mixed_config = make_config()
    mixed_config.topology_id = "A7-Z-POST"
    mixed_config.site_gate = None
    mixed_config.site_gates = {
        site: {
            "operator": "one_sided_threshold" if site in {"a", "m", "h", "z"} else "symmetric_threshold",
            "kappa": 0.0,
        }
        for site in ("a", "m", "h", "q_post", "k_post", "v", "z")
    }
    mixed = apply_activation_topology(GPTNeoXForCausalLM(mixed_config), torch=torch)
    mixed.load_state_dict(baseline.state_dict())
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    baseline_output = baseline(input_ids=input_ids, labels=input_ids)
    mixed_output = mixed(input_ids=input_ids, labels=input_ids)
    assert torch.equal(baseline_output.logits, mixed_output.logits)
    assert torch.equal(baseline_output.loss, mixed_output.loss)
    baseline_output.loss.backward()
    mixed_output.loss.backward()
    for (baseline_name, baseline_parameter), (mixed_name, mixed_parameter) in zip(
        baseline.named_parameters(), mixed.named_parameters(), strict=True
    ):
        assert baseline_name == mixed_name
        assert torch.equal(baseline_parameter.grad, mixed_parameter.grad)
