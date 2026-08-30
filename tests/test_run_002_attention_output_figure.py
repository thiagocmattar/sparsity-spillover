import sys
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "002-2026-08-29-l1n-spillover-local"
sys.path.insert(0, str(RUN_DIR))

import attention_output_figure  # noqa: E402


class _Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = torch.nn.Module()
        self.attention.dense = torch.nn.Linear(4, 4, bias=False)
        self.post_attention_dropout = torch.nn.Dropout(0.0)

    def forward(self, value):
        return self.post_attention_dropout(self.attention.dense(value)) + value


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer() for _ in range(6)])
        self.gpt_neox = SimpleNamespace(layers=self.layers)

    def forward(self, value):
        for layer in self.layers:
            value = layer(value)
        return value


def _statistics(site_values):
    return {
        "pooled_by_site": [
            {
                "name": site,
                "total": total,
                "threshold_hits": {"0.001": hits},
                "threshold_fractions": {"0.001": hits / total},
            }
            for site, (hits, total) in site_values.items()
        ],
        "rows": [],
    }


def test_capture_brackets_zero_dropout_after_wo_and_before_residual():
    model = _Model()
    with attention_output_figure.AttentionOutputCapture(model) as capture:
        model(torch.ones(2, 3, 4))
        assert sorted(capture.after_wo) == [
            f"attention_output.layer_{index}" for index in range(6)
        ]
        assert set(capture.after_wo) == set(capture.before_residual)
        for name in capture.after_wo:
            assert torch.equal(capture.after_wo[name], capture.before_residual[name])


def test_reduction_builds_confirmed_gelu_and_relu_five_point_lines():
    conditions = []
    diagnostic_rows = []
    original = {}
    order = 0
    for weight_index, weight in enumerate((0.0, 0.1, 0.5, 1.0, 5.0)):
        for activation in ("gelu", "relu"):
            order += 1
            attempt_id = f"attempt-{order}"
            control = weight_index == 0
            condition = {
                "id": f"{activation}-{'control' if control else weight}",
                "activation": activation,
                "order": order,
                "is_control": control,
                "pressure_weight": weight,
            }
            conditions.append(
                {
                    "attempt_id": attempt_id,
                    "condition": condition,
                    "final_validation_loss": 5.0,
                }
            )
            original[attempt_id] = _statistics({"h": (10 + order, 100)})
            diagnostic_rows.append(
                {
                    "attempt_id": attempt_id,
                    "statistics": _statistics({"attention_output": (20 + order, 1000)}),
                }
            )
    verification = {"status": "verified", "evidence_label": "valid", "conditions": conditions}
    diagnostic = {"status": "verified", "epsilon": 0.001, "conditions": diagnostic_rows}
    grouped = attention_output_figure.reduce_attention_output_points(
        verification, diagnostic, lambda attempt_id: original[attempt_id]
    )
    assert list(grouped) == ["gelu", "relu"]
    assert len(grouped["gelu"]) == len(grouped["relu"]) == 5
    assert grouped["gelu"][0]["label"] == "control"
    assert grouped["relu"][-1]["label"] == "lambda=5"
    assert grouped["gelu"][0]["h_near_zero"]["hits"] == 11
