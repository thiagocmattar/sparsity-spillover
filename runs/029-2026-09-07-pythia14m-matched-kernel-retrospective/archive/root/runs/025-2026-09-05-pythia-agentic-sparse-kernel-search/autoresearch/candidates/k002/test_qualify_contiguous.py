"""CPU-only checks for the append-only full-length qualification fixtures."""
import importlib.util
from pathlib import Path

import torch

SPEC = importlib.util.spec_from_file_location("run025_k002_contiguous_fixture_test", Path(__file__).with_name("qualify_contiguous.py"))
QUALIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUALIFY)


def test_strided_reference_inputs_preserve_values_but_change_layout():
    source = torch.randn(1, 4, 2048, 32).bfloat16()
    strided = QUALIFY.strided_copy(source)
    assert strided.stride(-1) == 2
    assert torch.equal(source, strided)
    assert source.data_ptr() != strided.data_ptr()


def test_changed_alternating_tiles_really_changes_gpu_branch_pattern():
    generator = torch.Generator().manual_seed(2502)
    first = QUALIFY.make_inputs(4, 32, "alternating_zero_tiles", 0, generator)
    generator.manual_seed(2502)
    changed = QUALIFY.make_inputs(4, 32, "alternating_zero_tiles", 1, generator)
    assert QUALIFY.expected_fast_tiles(first[0], 16).sum() == 4 * 64
    assert QUALIFY.expected_fast_tiles(changed[0], 16).sum() == 0
    assert not torch.equal(first[2], changed[2])


def test_allzero_and_single_active_query_fixtures_have_exact_declared_coverage():
    generator = torch.Generator().manual_seed(2502)
    zero, _, _ = QUALIFY.make_inputs(8, 64, "all_zero_q", 0, generator)
    assert QUALIFY.expected_fast_tiles(zero, 16).sum() == 8 * 128
    single, _, _ = QUALIFY.make_inputs(8, 64, "single_active_row", 0, generator)
    assert QUALIFY.expected_fast_tiles(single, 16).sum() == 8 * 127
    assert QUALIFY.expected_fast_tiles(single, 32).sum() == 8 * 63
