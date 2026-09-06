"""CPU helper tests, explicitly not a CUDA/model qualification."""
import importlib.util
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location("run025_diagnose_layers", Path(__file__).with_name("diagnose_layers.py"))
diagnose = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnose)


def test_comparison_reports_fixed_tolerance_failure():
    reference = torch.tensor([0., 1., -1.], dtype=torch.bfloat16)
    actual = torch.tensor([.5, 1., -1.], dtype=torch.bfloat16)
    result = diagnose.compare(reference, actual, {"relative_l2": .02, "atol": .25, "rtol": .02})
    assert not result["pass"] and result["violating_elements"] == 1
    assert result["unequal_elements"] == 1 and result["maximum_tolerance_excess"] == .25


def test_selected_fp64_dots_include_signed_operands_and_bias():
    x = torch.tensor([[[1., -2., 0.], [3., 0., -1.]]], dtype=torch.bfloat16)
    weight = torch.tensor([[1., 2., 3.], [-1., 1., 2.]], dtype=torch.bfloat16)
    bias = torch.tensor([.5, -.5], dtype=torch.bfloat16)
    reference = torch.nn.functional.linear(x, weight, bias)
    perturbed = reference.clone()
    perturbed[0, 1, 0] += 1
    rows = diagnose.fp64_samples(x, weight, bias, reference, perturbed, reference)
    assert len(rows) == 1
    assert rows[0]["flat_index"] == 2 and rows[0]["row"] == 1 and rows[0]["column"] == 0
    assert rows[0]["fp64_dot_with_bias"] == .5
    assert rows[0]["fp64_rounded_bf16"] == rows[0]["native"] == .5
    assert rows[0]["p0"] == 1.5
    assert diagnose.fp64_samples(x, weight, bias, reference, reference, reference) == []


def test_oracle_handles_bias_and_no_bias_without_mutating_input():
    x = torch.tensor([[1., -2.]], dtype=torch.bfloat16)
    weight_t = torch.tensor([[2.], [3.]], dtype=torch.bfloat16)
    output = torch.empty((1, 1), dtype=torch.bfloat16)
    original = x.clone()
    diagnose.fused_oracle(x, weight_t, torch.empty(0, dtype=torch.bfloat16), None, output)
    assert output.item() == -4
    diagnose.fused_oracle(x, weight_t, torch.tensor([.5], dtype=torch.bfloat16), None, output)
    assert output.item() == -3.5 and torch.equal(x, original)
