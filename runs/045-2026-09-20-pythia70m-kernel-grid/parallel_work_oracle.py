"""Actual extra scalar products from the row prepass in mixed fallback groups."""
def extra_scalar_products(h, z, *, rows=8, limit=8, fast_weights=True, skip=True):
    if not fast_weights or not skip:
        return [0, 0]
    tensors = [value.reshape(-1, value.shape[-1]) for value in (h, z)]
    nnz = [(value != 0).sum(-1) for value in tensors]
    safe = [((value == 0) | ((value.abs() >= 2.**-50) & (value.abs() <= 2.**50))).all(-1) for value in tensors]
    short = safe[0] & safe[1] & (nnz[0] <= limit) & (nnz[1] <= limit)
    fallback = (~short.reshape(-1, rows).all(-1)).repeat_interleave(rows)
    duplicate = short & fallback
    return [int(value[duplicate].sum()) * 512 for value in nnz]
