"""The counter oracle distinguishes useful scalar work from duplicate execution."""
import torch
from parallel_work_oracle import extra_scalar_products


def test_only_short_rows_in_mixed_groups_are_recomputed():
    h, z = torch.zeros(16, 2048), torch.zeros(16, 512)
    h[:, :2] = 1
    z[:, :1] = 1
    assert extra_scalar_products(h, z) == [0, 0]
    h[0, :9] = 1
    assert extra_scalar_products(h, z) == [7*2*512, 7*512]
    assert extra_scalar_products(h, z, skip=False) == [0, 0]
    assert extra_scalar_products(h, z, fast_weights=False) == [0, 0]


def test_unsafe_rows_trigger_fallback_but_do_not_run_the_prepass_outputs():
    h, z = torch.zeros(8, 2048), torch.zeros(8, 512)
    h[:, 0] = 1
    z[:, 0] = 1
    z[0, 0] = float('inf')
    assert extra_scalar_products(h, z) == [7*512, 7*512]
