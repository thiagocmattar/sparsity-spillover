from types import SimpleNamespace

import pytest
import torch

from sparsity_research.capture import clip_tensor
from sparsity_research.pythia import apply_activation_topology, topology_metadata
from sparsity_research.sites import (
    FixedOneSidedThreshold,
    FixedSymmetricThreshold,
    TOPOLOGIES,
    resolve_topology_and_gate,
)


def test_topology_registry_preserves_exact_ports():
    assert TOPOLOGIES["A0"].active_sites == ()
    assert TOPOLOGIES["A2"].active_sites == ("m", "h")
    assert TOPOLOGIES["A5-QK-PRE"].active_sites[-2:] == ("q_pre", "k_pre")
    assert TOPOLOGIES["A6-POST"].active_sites[-3:] == ("q_post", "k_post", "v")


def test_one_sided_threshold_keeps_equality_and_blocks_rejected_gradients():
    value = torch.tensor([-2.0, 0.5, 1.0, 2.0], requires_grad=True)
    output = FixedOneSidedThreshold(1.0)(value)
    assert output.tolist() == [0.0, 0.0, 1.0, 2.0]
    output.sum().backward()
    assert value.grad.tolist() == [0.0, 0.0, 1.0, 1.0]


def test_symmetric_threshold_keeps_signed_equality():
    value = torch.tensor([-1.0, -0.5, 0.5, 1.0], requires_grad=True)
    output = FixedSymmetricThreshold(1.0)(value)
    assert output.tolist() == [-1.0, 0.0, 0.0, 1.0]
    output.sum().backward()
    assert value.grad.tolist() == [1.0, 0.0, 0.0, 1.0]


def test_clipping_threshold_zeros_equality_unlike_architecture_gate():
    value = torch.tensor([-1.0, -0.5, 0.5, 1.0])
    clipped = clip_tensor(value, {"mode": "threshold", "threshold": 0.5}, torch=torch)
    assert clipped.tolist() == [-1.0, 0.0, 0.0, 1.0]


def test_topology_gate_contract_is_separate():
    topology, gate = resolve_topology_and_gate("A1-H", {"operator": "relu"})
    assert topology.active_sites == ("h",)
    assert gate == {"operator": "relu"}
    with pytest.raises(ValueError, match="A0"):
        resolve_topology_and_gate("A0", {"operator": "relu"})


class _FakeAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query_key_value = torch.nn.Linear(4, 12)
        self.dense = torch.nn.Linear(4, 4)
        self.head_size = 2
        self.config = SimpleNamespace()


class _FakeLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = torch.nn.Identity()
        self.post_attention_layernorm = torch.nn.Identity()
        self.mlp = torch.nn.Module()
        self.mlp.act = torch.nn.GELU()
        self.attention = _FakeAttention()


class _FakeModel(torch.nn.Module):
    def __init__(self, topology_id, site_gate):
        super().__init__()
        self.config = SimpleNamespace(topology_id=topology_id, site_gate=site_gate)
        self.gpt_neox = torch.nn.Module()
        self.gpt_neox.layers = torch.nn.ModuleList([_FakeLayer(), _FakeLayer()])


def test_model_realizes_gate_at_every_declared_site_and_layer():
    model = _FakeModel("A6-POST", {"operator": "symmetric_threshold", "kappa": 0.2})
    apply_activation_topology(model, torch=torch)
    metadata = topology_metadata(model)
    assert metadata["active_sites"] == ["a", "m", "h", "q_post", "k_post", "v"]
    for layer in model.gpt_neox.layers:
        assert isinstance(layer.a_gate, FixedSymmetricThreshold)
        assert isinstance(layer.m_gate, FixedSymmetricThreshold)
        assert isinstance(layer.mlp.act, FixedSymmetricThreshold)
        assert isinstance(layer.attention.q_post_gate, FixedSymmetricThreshold)
        assert not hasattr(layer.attention, "q_pre_gate")

