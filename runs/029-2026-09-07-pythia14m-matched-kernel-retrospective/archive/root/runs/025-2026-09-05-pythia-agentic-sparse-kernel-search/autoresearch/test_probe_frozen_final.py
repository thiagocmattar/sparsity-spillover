import importlib.util
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "run025_probe_frozen_final", HERE / "probe_frozen_final.py"
)
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)


def freeze(size):
    return {
        "selection_evidence": {
            "development_evaluation": [{"condition": f"{size}/a0"}]
        },
        "heldout_boundary": {"conditions": [f"{size}/a4-0p01"]},
    }


def manifest(size):
    return {
        "checkpoints": [
            {"id": f"{size}/a0", "size": size, "partition": "development"},
            {"id": f"{size}/a4-0p01", "size": size, "partition": "untuned"},
        ]
    }


def test_condition_boundary_requires_matching_size_partition_and_freeze():
    assert P.condition_row(manifest("70m"), "70m/a0", "k009", freeze("70m"))["partition"] == "development"
    assert P.condition_row(manifest("70m"), "70m/a4-0p01", "k009", freeze("70m"))["partition"] == "untuned"
    with pytest.raises(ValueError):
        P.condition_row(manifest("70m"), "70m/a0", "k010", freeze("70m"))


def test_site_policy_is_fixed_for_k010_and_topology_based_elsewhere():
    topology = {"active_sites": ["a", "m", "h", "q_post", "k_post", "v", "z"]}
    assert P.select_sites("active", topology, "k010") == {"z"}
    assert P.select_sites("active", topology, "k009") == {"a", "m", "h", "z"}
    assert P.select_sites("h", topology, "k012") == {"h"}
    with pytest.raises(ValueError):
        P.select_sites("all", topology, "k010")


def test_final_evaluator_has_exact_architecture_mapping():
    assert {key: value["size"] for key, value in P.SPECS.items()} == {
        "k009": "70m",
        "k010": "410m",
        "k012": "14m",
    }
