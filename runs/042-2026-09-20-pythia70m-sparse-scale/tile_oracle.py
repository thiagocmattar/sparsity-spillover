"""Independent occupancy accounting for M8/M16 h/z matrix instructions."""


def counts(value, rows, fast_weights=True, skip=True):
    import torch
    flat = value.reshape(-1, value.shape[-1])
    assert rows in (8, 16) and flat.shape[0] % rows == 0 and flat.shape[1] % 16 == 0
    groups, ktiles = flat.shape[0] // rows, flat.shape[1] // 16
    potential = groups * ktiles * (512 // 8)
    if not skip:
        return [potential, 0, 0]
    nonzero = flat != 0
    nnz = nonzero.sum(-1)
    safe = ((flat == 0) | ((flat.abs() >= 2.**-50) & (flat.abs() <= 2.**50))).all(-1)
    short = (nnz <= 2) & safe & fast_weights
    active = nonzero & ~short[:, None]
    active_tiles = active.reshape(groups, rows, ktiles, 16).any(1).any(-1).sum()
    issued = int(active_tiles) * (512 // 8)
    scalar = int(nnz[short].sum()) * 512
    return [issued, potential-issued, scalar]
