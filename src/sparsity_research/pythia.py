"""Random Pythia construction and exact canonical gate placement.

The attention forward mirrors Transformers 5.12 GPT-NeoX. Treat a Transformers
upgrade as a scientific/code change and re-run the placement and round-trip tests.
"""

from __future__ import annotations

from pathlib import Path
from types import MethodType
from typing import Any

from .sites import SITE_ORDER, build_gate, gate_metadata, resolve_topology_and_gates


def build_random_pythia(
    model_config: dict[str, Any],
    *,
    device: Any,
    torch: Any,
    auto_config: Any,
    auto_model: Any,
) -> Any:
    """Build FP32 random parameters from a pinned architecture config."""

    if model_config.get("initialization") != "random":
        raise ValueError("Pretraining requires model.initialization: random.")
    architecture = auto_config.from_pretrained(
        model_config["architecture"],
        revision=model_config["revision"],
    )
    topology, gates = resolve_topology_and_gates(
        model_config.get("topology_id"),
        model_config.get("site_gate"),
        model_config.get("site_gates"),
    )
    architecture.topology_id = topology.topology_id
    architecture.site_gate = (
        None if model_config.get("site_gates") is not None else model_config.get("site_gate")
    )
    if model_config.get("site_gates") is not None:
        architecture.site_gates = {site: dict(spec) for site, spec in gates.items()}
    architecture.torch_dtype = torch.float32
    model = auto_model.from_config(architecture)
    apply_activation_topology(model, torch=torch)
    return model.to(device=device, dtype=torch.float32)


def load_checkpoint_pythia(auto_model: Any, checkpoint: str | Path, *, torch: Any) -> Any:
    model = auto_model.from_pretrained(checkpoint)
    return apply_activation_topology(model, torch=torch)


def apply_activation_topology(model: Any, *, torch: Any) -> Any:
    config = getattr(model, "config", None)
    topology, gates = resolve_topology_and_gates(
        getattr(config, "topology_id", None),
        getattr(config, "site_gate", None),
        getattr(config, "site_gates", None),
    )
    if getattr(model, "_sparsity_topology_applied", False):
        return model
    layers = getattr(getattr(model, "gpt_neox", None), "layers", None)
    if layers is None:
        raise ValueError("Canonical topology placement requires GPT-NeoX layers.")
    active = frozenset(topology.active_sites)
    for index, layer in enumerate(layers):
        _validate_layer(layer, index=index, active=active)
    for layer in layers:
        if "a" in active:
            layer.a_gate = build_gate(gates["a"], torch_module=torch)
            layer.input_layernorm.register_forward_hook(_gate_output_hook(layer.a_gate))
        if "m" in active:
            layer.m_gate = build_gate(gates["m"], torch_module=torch)
            layer.post_attention_layernorm.register_forward_hook(_gate_output_hook(layer.m_gate))
        if "h" in active:
            layer.mlp.act = build_gate(gates["h"], torch_module=torch)
        if active.intersection(_ATTENTION_SITES):
            _install_attention_ports(layer.attention, active=active, gates=gates, torch=torch)
    model._sparsity_topology_applied = True
    return model


def expose_attention_sites(model: Any, *, torch: Any) -> Any:
    """Expose identity taps for attention capture without activating gates."""

    layers = getattr(getattr(model, "gpt_neox", None), "layers", None)
    if layers is None:
        raise ValueError("Attention capture requires GPT-NeoX layers.")
    for index, layer in enumerate(layers):
        attention = getattr(layer, "attention", None)
        if attention is None:
            raise ValueError(f"Could not resolve attention in layer {index}.")
        if all(getattr(attention, f"{site}_site", None) is not None for site in _ATTENTION_SITES):
            continue
        _validate_attention(attention, index=index)
        _install_attention_ports(
            attention,
            active=frozenset(),
            gates={},
            torch=torch,
        )
    return model


def topology_metadata(model: Any) -> dict[str, Any]:
    """Verify every layer realizes the configured topology and gate."""

    config = getattr(model, "config", None)
    topology, expected_gates = resolve_topology_and_gates(
        getattr(config, "topology_id", None),
        getattr(config, "site_gate", None),
        getattr(config, "site_gates", None),
    )
    layers = getattr(getattr(model, "gpt_neox", None), "layers", None)
    if layers is None:
        raise ValueError("Topology inspection requires GPT-NeoX layers.")
    expected = frozenset(topology.active_sites)
    for index, layer in enumerate(layers):
        modules = {
            "a": getattr(layer, "a_gate", None),
            "m": getattr(layer, "m_gate", None),
            "h": getattr(getattr(layer, "mlp", None), "act", None),
        }
        attention = getattr(layer, "attention", None)
        for site in _ATTENTION_SITES:
            modules[site] = getattr(attention, f"{site}_gate", None)
        realized = frozenset(site for site, module in modules.items() if gate_metadata(module) is not None)
        if realized != expected:
            raise ValueError(
                f"Layer {index} realizes {sorted(realized)}, expected {list(topology.active_sites)}."
            )
        for site in SITE_ORDER:
            if site in expected and gate_metadata(modules[site]) != expected_gates[site]:
                raise ValueError(f"Layer {index} realized gate differs at site {site}.")
    metadata = {
        "topology_id": topology.topology_id,
        "active_sites": list(topology.active_sites),
        "qk_placement": topology.qk_placement,
    }
    configured_site_gates = getattr(config, "site_gates", None)
    if configured_site_gates is None:
        uniform = None if not expected_gates else dict(next(iter(expected_gates.values())))
        metadata["site_gate"] = uniform
    else:
        metadata["site_gate"] = None
        metadata["site_gates"] = {
            site: dict(expected_gates[site]) for site in topology.active_sites
        }
    return metadata


_ATTENTION_SITES = ("q_pre", "k_pre", "q_post", "k_post", "v", "z")


def _validate_layer(layer: Any, *, index: int, active: frozenset[str]) -> None:
    for site, path in (
        ("a", "input_layernorm"),
        ("m", "post_attention_layernorm"),
    ):
        if site in active and getattr(layer, path, None) is None:
            raise ValueError(f"Site {site} cannot resolve layer {index}.{path}.")
    if "h" in active and getattr(getattr(layer, "mlp", None), "act", None) is None:
        raise ValueError(f"Site h cannot resolve the MLP activation in layer {index}.")
    if active.intersection(_ATTENTION_SITES):
        attention = getattr(layer, "attention", None)
        if attention is None:
            raise ValueError(f"Attention sites cannot resolve layer {index}.attention.")
        _validate_attention(attention, index=index)


def _validate_attention(attention: Any, *, index: int) -> None:
    for attribute in ("query_key_value", "dense", "head_size", "config"):
        if not hasattr(attention, attribute):
            raise ValueError(f"Attention site placement requires {attribute} in layer {index}.")


def _install_attention_ports(
    attention: Any,
    *,
    active: frozenset[str],
    gates: dict[str, dict[str, Any]],
    torch: Any,
) -> None:
    for site in _ATTENTION_SITES:
        setattr(attention, f"{site}_site", torch.nn.Identity())
        if site in active:
            setattr(attention, f"{site}_gate", build_gate(gates[site], torch_module=torch))
    attention.forward = MethodType(_attention_forward, attention)


def _attention_forward(
    self: Any,
    hidden_states: Any,
    attention_mask: Any,
    layer_past: Any = None,
    position_embeddings: Any = None,
    **kwargs: Any,
) -> tuple[Any, Any]:
    from transformers.models.gpt_neox import modeling_gpt_neox

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, 3 * self.head_size)
    qkv = self.query_key_value(hidden_states).view(hidden_shape).transpose(1, 2)
    query, key, value = qkv.chunk(3, dim=-1)

    query = _optional_gate(self, "q_pre", query)
    key = _optional_gate(self, "k_pre", key)
    query = self.q_pre_site(query)
    key = self.k_pre_site(key)

    cos, sin = position_embeddings
    query, key = modeling_gpt_neox.apply_rotary_pos_emb(query, key, cos, sin)

    query = self.q_post_site(_optional_gate(self, "q_post", query))
    key = self.k_post_site(_optional_gate(self, "k_post", key))
    value = self.v_site(_optional_gate(self, "v", value))

    if layer_past is not None:
        key, value = layer_past.update(key, value, self.layer_idx)

    interface = modeling_gpt_neox.ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation,
        modeling_gpt_neox.eager_attention_forward,
    )
    output, weights = interface(
        self,
        query,
        key,
        value,
        attention_mask,
        scaling=self.scaling,
        dropout=0.0 if not self.training else self.attention_dropout,
        **kwargs,
    )
    output = output.reshape(*input_shape, -1).contiguous()
    output = self.z_site(_optional_gate(self, "z", output))
    output = self.dense(output)
    return output, weights


def _optional_gate(module: Any, site: str, value: Any) -> Any:
    gate = getattr(module, f"{site}_gate", None)
    return value if gate is None else gate(value)


def _gate_output_hook(gate: Any) -> Any:
    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        return gate(output)

    return hook
