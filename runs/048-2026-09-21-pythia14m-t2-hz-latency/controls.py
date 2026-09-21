"""Four causal execution modes; gates and all other implementation choices stay fixed."""
import math

MASKS = {'A': (True, True), 'B': (False, True),
         'C': (True, False), 'D': (False, False)}
MODES = tuple(MASKS)


def contrasts(latencies):
    if set(latencies) != set(MODES) or any(not math.isfinite(t) or t <= 0 for t in latencies.values()):
        raise ValueError('All four positive finite latencies are required')
    a, b, c, d = (latencies[m] for m in MODES)
    result = {}
    for name, off in [('h_given_z', b), ('z_given_h', c), ('joint_hz', d)]:
        result[name] = {'saved_ms': off-a, 'reduction_percent': 100*(off-a)/off,
                        'speedup': off/a}
    result['interaction_ms'] = d-b-c+a
    return result
