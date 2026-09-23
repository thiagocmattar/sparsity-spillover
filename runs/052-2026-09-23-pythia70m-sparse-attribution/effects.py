"""Paired process/input bootstrap for raw and dense-adjusted threshold gaps."""
import numpy as np


def estimands(milliseconds):
    """Arrays share shape [process,input]; selection of controls is external."""
    g={key:float(np.exp(np.mean(np.log(value)))) for key,value in milliseconds.items()}
    delta14=g['14_sparse05']-g['14_sparse10']
    delta70=g['70_sparse05']-g['70_sparse10']
    adjusted14=delta14-(g['14_dense05']-g['14_dense10'])
    adjusted70=delta70-(g['70_dense05']-g['70_dense10'])
    return {'delta14_ms':delta14,'delta70_ms':delta70,'delta70_minus14_ms':delta70-delta14,
            'adjusted14_ms':adjusted14,'adjusted70_ms':adjusted70,
            'adjusted70_minus14_ms':adjusted70-adjusted14,
            'gain70_05_ms':g['70_dense05']-g['70_sparse05'],
            'gain70_10_ms':g['70_dense10']-g['70_sparse10']}


def paired_intervals(milliseconds,differences,ratios=None,qualified=None,draws=10000,seed=2504):
    """Same resample for all backends/conditions; suppress unqualified intervals."""
    arrays={k:np.asarray(v,dtype=float) for k,v in milliseconds.items()}
    shapes={a.shape for a in arrays.values()}
    if len(shapes)!=1:raise ValueError('Unmatched process/input dimensions')
    shape=next(iter(shapes))
    if len(shape)!=2 or shape[0]<2 or shape[1]<2:raise ValueError('At least two paired processes and inputs required')
    if any(not np.all(np.isfinite(a)&(a>0)) for a in arrays.values()):raise ValueError('Invalid timing')
    if draws<2:raise ValueError('At least two bootstrap draws required')
    ratios=ratios or {};names=list(arrays);position={k:i for i,k in enumerate(names)}
    log=np.log(np.stack(list(arrays.values())))
    def contrast(g):
        result={name:sum(coef*g[position[key]] for key,coef in terms.items()) for name,terms in differences.items()}
        result.update({name:g[position[num]]/g[position[den]] for name,(num,den) in ratios.items()})
        return result
    point={k:float(v) for k,v in contrast(np.exp(log.mean(axis=(1,2)))).items()}
    rng=np.random.default_rng(seed);samples={k:[] for k in point}
    for start in range(0,draws,128):
        batch=min(128,draws-start)
        pi=rng.integers(shape[0],size=(batch,shape[0]));ii=rng.integers(shape[1],size=(batch,shape[1]))
        draw=contrast(np.exp(log[:,pi[:,:,None],ii[:,None,:]].mean(axis=(2,3))))
        for k,v in draw.items():samples[k].extend(v.tolist())
    required={**{k:list(v) for k,v in differences.items()},**{k:list(v) for k,v in ratios.items()}}
    accepted={k:qualified is None or all(qualified[n] for n in ns) for k,ns in required.items()}
    return {'point':point,'qualified':accepted,
            'crossed_process_input_95':{k:np.quantile(v,[.025,.975]).tolist() if accepted[k] else None for k,v in samples.items()},
            'scope':'Paired round and input resampling across conditions and sizes; no GPU-population or training-seed inference.'}


def interval(milliseconds,draws=10000,seed=2504,qualified=None):
    differences={
        'delta14_ms':{'14_sparse05':1,'14_sparse10':-1},
        'delta70_ms':{'70_sparse05':1,'70_sparse10':-1},
        'delta70_minus14_ms':{'70_sparse05':1,'70_sparse10':-1,'14_sparse05':-1,'14_sparse10':1},
        'adjusted14_ms':{'14_sparse05':1,'14_sparse10':-1,'14_dense05':-1,'14_dense10':1},
        'adjusted70_ms':{'70_sparse05':1,'70_sparse10':-1,'70_dense05':-1,'70_dense10':1},
        'adjusted70_minus14_ms':{'70_sparse05':1,'70_sparse10':-1,'70_dense05':-1,'70_dense10':1,'14_sparse05':-1,'14_sparse10':1,'14_dense05':1,'14_dense10':-1},
        'gain70_05_ms':{'70_dense05':1,'70_sparse05':-1},
        'gain70_10_ms':{'70_dense10':1,'70_sparse10':-1}}
    return paired_intervals(milliseconds,differences,draws=draws,seed=seed,qualified=qualified)
