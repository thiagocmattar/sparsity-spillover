import math

import pytest
import torch

from sparsity_research.optimization import learning_rate, warmup_steps
from sparsity_research.pressure import (
    activation_l1,
    apply_ol1_correction,
    gradient_interaction,
    parse_pressure_config,
)


def test_activation_l1_weights_captured_tensors_equally():
    loss = activation_l1(
        {
            "narrow": torch.tensor([2.0]),
            "wide": torch.zeros(100),
        }
    )
    assert float(loss) == 1.0


def test_gradient_interaction_reports_global_conflict():
    result = gradient_interaction(
        [torch.tensor([1.0, 0.0]), torch.tensor([2.0])],
        [torch.tensor([-2.0, 0.0]), torch.tensor([0.5])],
    )
    assert result["task_pressure_gradient_dot"] == -1.0
    assert result["gradient_conflict"] is True
    assert result["task_gradient_norm"] == pytest.approx(math.sqrt(5.0))


def test_pressure_config_keeps_methods_distinct():
    none = parse_pressure_config({"method": "none", "sites": [], "weight": 0.0})
    naive = parse_pressure_config({"method": "l1_naive", "sites": ["h"], "weight": 0.1})
    ol1 = parse_pressure_config(
        {"method": "orthogonal_l1", "sites": ["h"], "weight": 0.1, "step_budget": 0.2}
    )
    assert not none.enabled
    assert naive.enabled and not naive.orthogonal
    assert ol1.orthogonal


def test_ol1_projects_conflict_and_respects_trust_budget():
    parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.1, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    parameter.grad = torch.tensor([1.0, 0.0])
    task_grads = [parameter.grad.detach().clone()]
    optimizer.step()
    before = parameter.detach().clone()
    result = apply_ol1_correction(
        optimizer,
        [parameter],
        task_grads,
        [torch.tensor([-1.0, 1.0])],
        pressure_weight=1.0,
        step_budget=0.1,
    )
    assert result["projection_applied"] is True
    assert result["task_pressure_dot_after"] == pytest.approx(0.0, abs=1e-6)
    assert result["pressure_to_task_ratio_final"] <= 0.1 + 1e-9
    assert not torch.equal(parameter.detach(), before)


def test_learning_rate_schedule_hits_peak_and_final_ratio():
    assert warmup_steps(101) == 2
    assert learning_rate(2, peak=1.0, max_steps=101, warmup=2) == 1.0
    assert learning_rate(101, peak=1.0, max_steps=101, warmup=2) == 0.1

