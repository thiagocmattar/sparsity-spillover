"""Approved factorial contrasts in full-model milliseconds."""
import math

MODES = ('frozen', 't00', 't10', 't01', 't11')


def mask(mode):
    if mode not in MODES:
        raise ValueError(f'Unknown mechanism mode: {mode}')
    if mode == 'frozen':
        return {'tile': True, 'short': True}
    return {'tile': mode[1] == '1', 'short': mode[2] == '1'}


def contrasts(times):
    if not set(MODES) <= times.keys() or any(
            not math.isfinite(times[m]) or times[m] <= 0 for m in MODES):
        raise ValueError('Five finite, positive mode latencies required')
    a, b, c, d = [times[m] for m in ('t00', 't10', 't01', 't11')]
    return {'tile_given_short_ms': c-d, 'short_given_tile_ms': b-d,
            'joint_ms': a-d, 'tile_alone_ms': a-b, 'short_alone_ms': a-c,
            'interaction_ms': b+c-a-d}


def difference_spans(ranges):
    """Conservative extrema, not confidence intervals or paired-run estimates."""
    coefficients = {
        'tile_given_short_ms': {'t01': 1, 't11': -1},
        'short_given_tile_ms': {'t10': 1, 't11': -1},
        'joint_ms': {'t00': 1, 't11': -1},
        'tile_alone_ms': {'t00': 1, 't10': -1},
        'short_alone_ms': {'t00': 1, 't01': -1},
        'interaction_ms': {'t10': 1, 't01': 1, 't00': -1, 't11': -1}}
    return {name: [sum(sign*ranges[m][0 if sign > 0 else 1] for m, sign in terms.items()),
                   sum(sign*ranges[m][1 if sign > 0 else 0] for m, sign in terms.items())]
            for name, terms in coefficients.items()}
