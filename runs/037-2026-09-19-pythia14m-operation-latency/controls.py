"""The approved fixed-checkpoint execution masks; gates never change."""
OPS = ('a', 'm', 'h', 'z', 'qk', 'pv')
MODES = ('frozen', 'full', 'off', 'projection') + tuple('without-' + op for op in OPS)


def mask(mode):
    if mode not in MODES:
        raise ValueError(f'Unknown mode: {mode}')
    return {op: mode != 'off' and (mode != 'projection' or op not in ('qk', 'pv'))
            and mode != 'without-' + op for op in OPS}


def contrasts(latencies):
    """Conditional effects at full mode; these are not additive allocations."""
    import math
    if set(latencies) != set(MODES) or any(not math.isfinite(t) or t <= 0 for t in latencies.values()):
        raise ValueError('All ten positive finite mode latencies required')
    full = latencies['full']
    return {op: {'saved_ms': latencies['without-' + op] - full,
                 'speedup': latencies['without-' + op] / full} for op in OPS}
