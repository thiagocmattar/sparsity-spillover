"""Approved fixed-checkpoint implementation comparisons."""
MODES = ('frozen', 'port-a', 'port-m', 'port-am', 'port-am-dense')


def settings(mode):
    if mode not in MODES:
        raise ValueError(mode)
    return {'a': mode in ('port-a', 'port-am', 'port-am-dense'),
            'm': mode in ('port-m', 'port-am', 'port-am-dense'),
            'skip': mode != 'port-am-dense'}


def contrasts(latencies):
    import math
    if set(latencies) != set(MODES) or any(not math.isfinite(x) or x <= 0 for x in latencies.values()):
        raise ValueError('Five positive finite mode latencies required')
    result = {mode: {'saved_ms': latencies['frozen'] - latencies[mode],
                     'speedup': latencies['frozen'] / latencies[mode]} for mode in MODES[1:]}
    result['port_sparse_effect'] = {'saved_ms': latencies['port-am-dense'] - latencies['port-am'],
                                   'speedup': latencies['port-am-dense'] / latencies['port-am']}
    return result
