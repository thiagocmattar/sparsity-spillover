"""A CPU model of ordered RN Kahan arithmetic, not an executed CUDA result."""
from pathlib import Path

import numpy as np
import torch


def ordered_dot(x, weight, *, compensate):
    assert x.dtype == weight.dtype == torch.bfloat16
    total, correction = np.float32(0), np.float32(0)
    for value, coefficient in zip(x.float().numpy(), weight.float().numpy()):
        if value == 0:
            continue
        product = np.float32(value * coefficient)
        if compensate:
            adjusted = np.float32(product - correction)
            next_total = np.float32(total + adjusted)
            correction = np.float32(np.float32(next_total - total) - adjusted)
            total = next_total
        else:
            total = np.float32(total + product)
    return float(total)


def test_exact_bf16_products_can_be_lost_by_serial_fp32_sum():
    values = torch.tensor([65536.] + [1 / 512.] * 4096 + [-65536.], dtype=torch.bfloat16)
    weight = torch.ones_like(values)
    exact = float((values.double() * weight.double()).sum())
    assert exact == 8.
    assert ordered_dot(values, weight, compensate=False) == 0.
    assert ordered_dot(values, weight, compensate=True) == exact


def test_signed_sparse_typical_amplitudes_against_float64():
    generator = torch.Generator().manual_seed(2511)
    naive_error, compensated_error = [], []
    for scale in (.1, 1., 10.):
        for _ in range(8):
            values = (torch.randn(4096, generator=generator) * scale).bfloat16()
            values[torch.rand(4096, generator=generator) < .8] = 0
            weight = (torch.randn(4096, generator=generator) * .1).bfloat16()
            exact = float((values.double() * weight.double()).sum())
            naive_error.append(abs(ordered_dot(values, weight, compensate=False) - exact))
            compensated_error.append(abs(ordered_dot(values, weight, compensate=True) - exact))
    assert sum(compensated_error) < sum(naive_error)


def test_cuda_explicit_rounding_and_same_fused_bias_contract():
    source = Path(__file__).with_name("kernel.cu").read_text()
    assert "const float product = __fmul_rn(x, w)" in source
    assert "const float adjusted = __fsub_rn(product, correction)" in source
    assert "const float next = __fadd_rn(sum, adjusted)" in source
    assert "correction = __fsub_rn(__fsub_rn(next, sum), adjusted)" in source
    assert "float correction[8] = {0.f}" in source
    assert "const float result = __fadd_rn(acc[j]," in source
    assert "__float2bfloat16_rn(result)" in source
    assert "fmaf(" not in source
