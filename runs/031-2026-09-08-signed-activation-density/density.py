"""Count-first signed histograms for the approved seven-checkpoint diagnostic."""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

SITES = ('h', 'm', 'q_post', 'k_post', 'v')
GROUPS = {'FFN activations': ('h', 'm'),
          'Attention activations': ('q_post', 'k_post', 'v')}
GRID = {'lower': -8., 'upper': 8., 'bins': 16000, 'edges_dtype': 'float32',
        'intervals': '[left,right), final bin includes right edge',
        'zero': 'excluded from histogram; retained as exact point mass'}


def edges():
    return np.linspace(GRID['lower'], GRID['upper'], GRID['bins']+1, dtype=np.float32)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    payload = (json.dumps(data, sort_keys=True, allow_nan=False)+'\n').encode()
    temporary.write_bytes(gzip.compress(payload, mtime=0) if path.suffix=='.gz' else payload)
    temporary.replace(path)


def read_json(path):
    path = Path(path)
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix=='.gz' else raw)


class SignedHistogram:
    """Accumulate integer CUDA/CPU counts without moving each histogram to host."""
    def __init__(self, *, torch, device):
        self.torch = torch
        self.edges = torch.as_tensor(edges(), device=device)
        self.counts, self.extrema, self.dtypes = {}, {}, {}

    def update(self, activations):
        torch = self.torch
        for name, value in activations.items():
            flat = value.detach().float().reshape(-1)
            finite = flat[torch.isfinite(flat)]
            self.dtypes.setdefault(name, set()).add(str(value.dtype))
            if not finite.numel():
                raise ValueError(f'No finite activations at {name}')
            limits = torch.stack((finite.min(), finite.max()))
            if name in self.extrema:
                previous = self.extrema[name]
                limits = torch.stack((torch.minimum(previous[0], limits[0]),
                                      torch.maximum(previous[1], limits[1])))
            self.extrema[name] = limits
            nonzero = finite[finite != 0]
            # Index 0 is underflow; the last index is overflow.
            index = torch.bucketize(nonzero, self.edges, right=True)
            index[nonzero == self.edges[-1]] = len(self.edges)-1
            counts = torch.bincount(index, minlength=len(self.edges)+1)
            if name not in self.counts:
                self.counts[name] = counts
            else:
                self.counts[name] += counts

    def rows(self, moment_rows):
        if set(self.counts) != {row['name'] for row in moment_rows}:
            raise ValueError('Histogram and moment sites differ')
        result = []
        for moments in moment_rows:
            name = moments['name']
            counts = self.counts[name].cpu().tolist()
            low, high = self.extrema[name].cpu().tolist()
            row = dict(moments, histogram=counts[1:-1], underflow=counts[0],
                       overflow=counts[-1], minimum=low, maximum=high,
                       dtypes=sorted(self.dtypes[name]))
            validate_row(row)
            result.append(row)
        return result


def validate_row(row):
    counts = row['histogram']
    if len(counts) != GRID['bins'] or any(type(n) is not int or n < 0 for n in counts):
        raise ValueError('Invalid integer histogram')
    if sum(counts)+row['underflow']+row['overflow']+row['exact_zero_count'] != row['finite']:
        raise ValueError('Histogram, tail and zero counts do not partition finite values')
    if row['finite']+row['nonfinite'] != row['total'] or row['nonfinite']:
        raise ValueError('Nonfinite activations or inconsistent total')


def pool(rows, name, sites):
    chosen = [r for r in rows if r['name'].split('.layer_')[0] in sites]
    if {r['name'].split('.layer_')[0] for r in chosen} != set(sites):
        raise ValueError(f'Missing sites for {name}')
    for row in chosen:
        validate_row(row)
    result = {'name': name, 'sites': list(sites),
              'histogram': [sum(values) for values in zip(*(r['histogram'] for r in chosen))],
              **{key: sum(r[key] for r in chosen) for key in
                 ('total','finite','nonfinite','exact_zero_count','underflow','overflow',
                  'sum','square_sum','absolute_sum')},
              'threshold_hits': {eps: sum(r['threshold_hits'][eps] for r in chosen)
                                 for eps in chosen[0]['threshold_hits']},
              'minimum': min(r['minimum'] for r in chosen),
              'maximum': max(r['maximum'] for r in chosen)}
    result['rms'] = (result['square_sum']/result['finite'])**.5
    result['l2_norm'] = result['square_sum']**.5
    result['site_element_weights'] = {
        site: sum(r['total'] for r in chosen if r['name'].split('.layer_')[0]==site)/result['total']
        for site in sites}
    validate_row(result)
    return result


def nonzero_density(row):
    validate_row(row)
    return np.asarray(row['histogram'], dtype=np.float64)/(row['total']*np.diff(edges().astype(np.float64)))
