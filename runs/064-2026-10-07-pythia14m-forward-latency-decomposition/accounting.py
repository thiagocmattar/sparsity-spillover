"""Count-derived, additive latency budgets and trace unions."""
import math


def intervals_union(rows):
    total=0.;end=None
    for start,stop in sorted(rows):
        if stop<start: raise ValueError('Reversed interval')
        if end is None or start>end: total+=stop-start
        elif stop>end: total+=stop-end
        end=stop if end is None else max(end,stop)
    return total


def budget(names, samples):
    if len(set(names))!=len(names) or not samples: raise ValueError('Unique stages and samples required')
    if any(len(row)!=len(names) or any(not math.isfinite(v) or v<0 for v in row) for row in samples):
        raise ValueError('Finite nonnegative complete stage timings required')
    means=[sum(row[i] for row in samples)/len(samples) for i in range(len(names))]
    total=sum(means)
    return {'total_us':total,'stages':[{'stage':name,'mean_us':value,'fraction':value/total if total else 0.} for name,value in zip(names,means)]}
