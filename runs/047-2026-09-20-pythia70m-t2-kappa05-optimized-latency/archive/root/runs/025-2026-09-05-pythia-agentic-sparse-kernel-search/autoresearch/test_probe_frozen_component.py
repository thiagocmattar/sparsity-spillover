from __future__ import annotations

import pytest

from probe_frozen_component import component_sites, frozen_allowed_sites


def topology(identifier: str, active: tuple[str, ...]) -> dict:
    return {"topology_id": identifier, "active_sites": list(active)}


def test_k013_respects_control_and_intervention_topology() -> None:
    assert frozen_allowed_sites("k013", topology("A0", ())) == {"a", "m", "h", "z"}
    assert component_sites("ffn", "k013", topology("A1-H", ("h",))) == {"h"}
    assert component_sites(
        "attention_projection", "k013", topology("A7-Z-POST", ("a", "m", "h", "z", "q", "k", "v"))
    ) == {"a", "z"}


def test_k016_uses_frozen_topology_dispatch_not_requested_gate_set() -> None:
    assert component_sites("ffn", "k016", topology("A7-Z-POST", ("a", "m", "h", "z"))) == {"h"}
    assert component_sites(
        "attention_projection", "k016", topology("A7-Z-POST", ("a", "m", "h", "z"))
    ) == {"z"}
    with pytest.raises(ValueError, match="has no ffn"):
        component_sites("ffn", "k016", topology("A0", ()))


def test_k010_has_projection_only() -> None:
    a7 = topology("A7-Z-POST", ("a", "m", "h", "z", "q", "k", "v"))
    assert component_sites("attention_projection", "k010", a7) == {"z"}
    with pytest.raises(ValueError, match="has no ffn"):
        component_sites("ffn", "k010", a7)


def test_unknown_component_or_implementation_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported component"):
        component_sites("qk", "k013", topology("A0", ()))
    with pytest.raises(ValueError, match="Unsupported frozen implementation"):
        frozen_allowed_sites("k999", topology("A0", ()))
