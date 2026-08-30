from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


SCRIPT = Path(__file__).with_name("01_evaluate.py")
SPEC = spec_from_file_location("analysis005_evaluate", SCRIPT)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_empirical_thresholds_use_zero_row_and_smallest_qualifying_order_statistic():
    result = MODULE.empirical_thresholds(
        np.array([0.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32),
        (0.0, 0.2, 0.4, 0.6, 1.0),
        np=np,
    )
    assert result["natural_exact_zero_count"] == 2
    assert [row["threshold"] for row in result["targets"]] == [0.0, 0.0, 0.0, 1.0, 3.0]
    assert [row["calibration_zero_count"] for row in result["targets"]] == [2, 2, 2, 3, 5]


def test_threshold_hooks_are_name_specific_and_clip_equality():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            layer = torch.nn.Module()
            layer.input_layernorm = torch.nn.Identity()
            layer.post_attention_layernorm = torch.nn.Identity()
            layer.mlp = torch.nn.Module()
            layer.mlp.act = torch.nn.Identity()
            layer.attention = torch.nn.Module()
            layer.attention.z_site = torch.nn.Identity()
            self.gpt_neox = SimpleNamespace(layers=[layer])

    model = Model()
    original_module_map = MODULE._module_map
    MODULE._module_map = lambda _model: {
        "a.layer_0": (model.gpt_neox.layers[0].input_layernorm, "output"),
        "m.layer_0": (model.gpt_neox.layers[0].post_attention_layernorm, "output"),
        "h.layer_0": (model.gpt_neox.layers[0].mlp.act, "output"),
        "z.layer_0": (model.gpt_neox.layers[0].attention.z_site, "input"),
    }
    thresholds = {
        "a.layer_0": 0.5,
        "m.layer_0": 1.0,
        "h.layer_0": 0.0,
        "z.layer_0": 2.0,
    }
    try:
        with MODULE.threshold_hooks(model, thresholds, torch=torch):
            value = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
            assert model.gpt_neox.layers[0].input_layernorm(value).tolist() == [
                -2.0,
                -1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                2.0,
            ]
            assert model.gpt_neox.layers[0].post_attention_layernorm(value).tolist() == [
                -2.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                2.0,
            ]
            assert model.gpt_neox.layers[0].attention.z_site(value).tolist() == [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
    finally:
        MODULE._module_map = original_module_map


def test_nondominated_prefers_lower_loss_and_higher_opportunity():
    def row(loss, opportunity):
        return {
            "validation": {"loss": loss},
            "logical_products": {"R_model": opportunity},
        }

    rows = [row(5.0, 0.1), row(5.1, 0.2), row(5.2, 0.15), row(4.9, 0.1)]
    assert MODULE.nondominated(rows) == [False, True, False, True]
