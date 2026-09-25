"""Untimed operand reference; no dependency on generated CUDA control code."""
import torch


def projection_work(value, *, tile, short, fast_weights=True):
    x = value.reshape(-1, value.shape[-1])
    m, k = x.shape
    if m == 0 or m % 8 or k not in (128, 512):
        raise ValueError('M8 rows and K128 or K512 required')
    live = x != 0
    nnz = live.sum(-1)
    safe = ((x == 0) | ((x.abs() >= 2.**-50) & (x.abs() <= 2.**50))).all(-1)
    eligible = (nnz <= 2) & safe & bool(fast_weights)
    routed = eligible & bool(short)
    completed = routed.reshape(-1, 8).all(1)
    original = live.reshape(m//8, 8, k//16, 16).any(1).any(-1)
    remaining = (live & ~routed[:, None]).reshape(m//8, 8, k//16, 16).any(1).any(-1)
    active = remaining if tile else (~completed[:, None]).expand(-1, k//16)
    potential = (m//8)*(k//16)*16
    issued = int(active.sum())*16
    group_omissions = int(completed.sum())*(k//16)*16
    scalar = int(nnz[routed].sum())*128
    stats = {
        'rows': m, 'projection_groups': m//8, 'tile_grid': (m//8)*(k//16),
        'eligible_rows': int(eligible.sum()), 'short_rows_executed': int(routed.sum()),
        'zero_rows_executed': int((routed & (nnz == 0)).sum()),
        'one_nonzero_rows_executed': int((routed & (nnz == 1)).sum()),
        'two_nonzero_rows_executed': int((routed & (nnz == 2)).sum()),
        'guard_rejected_short_rows': int(((nnz <= 2) & ~eligible).sum()),
        'original_empty_tiles': int((~original).sum()),
        'remaining_empty_tiles': int((~remaining).sum()),
        'newly_empty_tiles': int((original & ~remaining).sum()),
        'short_completed_groups': int(completed.sum()),
        'mma_potential': potential, 'mma_issued': issued,
        'mma_omitted_whole_short_group': group_omissions,
        'mma_omitted_empty_tile': potential-issued-group_omissions,
        'scalar_products': scalar,
        'source_weight_requests_bf16_elements': issued*128+scalar}
    return {'counts': stats, 'row_nnz_histogram': torch.bincount(nnz, minlength=k+1).cpu().tolist(),
            'completed_groups': completed,
            'legacy_counters': [issued, potential-issued, scalar]}


def same_bits(left, right):
    if left.dtype != torch.bfloat16 or right.dtype != torch.bfloat16:
        raise ValueError('BF16 operands required')
    return left.shape == right.shape and torch.equal(left.contiguous().view(torch.int16),
                                                    right.contiguous().view(torch.int16))
