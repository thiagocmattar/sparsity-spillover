"""K002 mathematical/contract tests; CUDA remains a separate qualification."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

SPEC = importlib.util.spec_from_file_location("run025_k002_attention_tested", Path(__file__).with_name("attention.py"))
K002 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = K002
SPEC.loader.exec_module(K002)


@pytest.mark.parametrize("pattern", ["zero_q", "zero_k", "zero_v", "signed", "mixed"])
def test_exact_zero_query_identity_matches_dense_causal_mathematics(pattern):
    generator = torch.Generator().manual_seed(2502)
    q, k, v = [torch.randn(1, 2, 17, 32, generator=generator).bfloat16() for _ in range(3)]
    if pattern == "zero_q":
        q.zero_()
    elif pattern == "zero_k":
        k.zero_()
    elif pattern == "zero_v":
        v.zero_()
    elif pattern == "mixed":
        q[:, :, :16] = 0
    expected = F.scaled_dot_product_attention(q.double(), k.double(), v.double(), is_causal=True)
    actual = K002.reference(q, k, v)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def test_zero_query_is_prefix_mean_not_zero_output_or_removed_key():
    q = torch.zeros(1, 1, 4, 32, dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    v[..., 0, :] = 4
    # Three zero V rows still contribute to the prefix softmax denominator.
    expected = torch.tensor([4., 2., 4. / 3., 1.]).view(1, 1, 4, 1).double().expand_as(v)
    torch.testing.assert_close(K002.reference(q, k, v), expected, atol=1e-7, rtol=1e-7)


def test_future_values_cannot_change_earlier_outputs_for_either_branch():
    generator = torch.Generator().manual_seed(2503)
    q, k, v = [torch.randn(1, 1, 17, 32, generator=generator).bfloat16() for _ in range(3)]
    q[..., :4, :] = 0
    original = K002.reference(q, k, v)
    changed = v.clone()
    changed[..., 8:, :] = 100
    torch.testing.assert_close(original[..., :8, :], K002.reference(q, k, changed)[..., :8, :], atol=0, rtol=0)


def test_unsupported_mask_interface_preserves_dense_result_and_layout():
    q = torch.randn(1, 2, 4, 32)
    k, v = torch.randn_like(q), torch.randn_like(q)
    mask = torch.ones(4, 4, dtype=torch.bool).tril()
    mask[:, 0] = False
    mask[0, 0] = True
    module = SimpleNamespace(is_causal=True)
    actual, weights = K002.interface(module, q, k, v, mask, scaling=.3)
    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=.3)
    assert weights is None
    assert actual.shape == (1, 4, 2, 32)
    torch.testing.assert_close(actual, expected.transpose(1, 2).contiguous(), atol=0, rtol=0)


def test_contract_rejects_wrong_precision_bad_tiles_and_cpu_kernel_substitute():
    q = torch.zeros(1, 1, 17, 32, dtype=torch.bfloat16)
    K002.validate_shape(q, q, q, 16)
    with pytest.raises(ValueError, match="BF16"):
        K002.validate_shape(q.float(), q.float(), q.float(), 16)
    with pytest.raises(ValueError, match="tile"):
        K002.validate_shape(q, q, q, 8)
    with pytest.raises(ValueError, match="CUDA"):
        K002.attention(q, q, q)


def test_workspace_bytes_include_prefix_output_metadata_and_flags():
    q = torch.zeros(1, 2, 129, 32, dtype=torch.bfloat16)
    workspace = K002.Workspace(q, 16)
    expected = q.numel() * 6 + 1 * 2 * 2 * 32 * 4 + 1 * 2 * 9
    assert workspace.bytes == expected
    assert workspace.nchunks == 2
    assert workspace.ntiles == 9
