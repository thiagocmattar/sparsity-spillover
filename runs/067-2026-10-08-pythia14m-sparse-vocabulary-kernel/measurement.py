"""Matched graph timing and count-first full-logit numerical qualification."""
import time
import numpy as np
import torch


class Graph:
    def __init__(self, function, inputs):
        self.inputs=[x.clone() for x in inputs]
        self.function=function
        start=time.monotonic()
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):self.output=function(*self.inputs)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):self.output=function(*self.inputs)
        self.graph.replay();torch.cuda.synchronize()
        self.setup_seconds=time.monotonic()-start

    def stage(self, inputs):
        for target,value in zip(self.inputs,inputs):target.copy_(value)

    def __call__(self):
        self.graph.replay()
        return self.output


def paired_timing(graphs, inputs, *, passes, seed, sink):
    rng=np.random.default_rng(seed)
    rows=[]
    for index,values in enumerate(inputs):
        for graph in graphs.values():graph.stage(values)
        for _ in range(3):
            for graph in graphs.values():graph()
        torch.cuda.synchronize()
        for repeat in range(passes):
            for mode in rng.permutation(list(graphs)):
                graph=graphs[mode]
                # Primary host latency has no CUDA timing events. Event timing uses a separate replay.
                torch.cuda.synchronize();wall=time.perf_counter_ns()
                graph();torch.cuda.synchronize()
                elapsed=(time.perf_counter_ns()-wall)/1e6
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record();graph();end.record();end.synchronize()
                row={'input':index,'pass':repeat,'mode':str(mode),'host_ms':elapsed,'cuda_ms':start.elapsed_time(end)}
                rows.append(row);sink(row)
    return rows


def summarize(samples):
    result={}
    for mode in sorted({r['mode'] for r in samples}):
        rows=[r for r in samples if r['mode']==mode]
        result[mode]={'count':len(rows)}
        for field in ('host_ms','cuda_ms'):
            values=np.array([r[field] for r in rows])
            result[mode][field]={'geomean':float(np.exp(np.log(values).mean())),
                                 'mean':float(values.mean()),'median':float(np.median(values)),
                                 'p05':float(np.quantile(values,.05)),'p95':float(np.quantile(values,.95))}
    return result


def compare(reference, actual, bounds):
    # Chunk by token positions so the checker does not allocate several full FP32 V matrices.
    maximum=0.; squared=0.; ref_squared=0.; finite=True; elementwise=True; exact=True
    for a,b in zip(actual.reshape(-1,actual.shape[-1]).split(64),reference.reshape(-1,reference.shape[-1]).split(64)):
        a,b=a.float(),b.float();difference=a-b
        finite=finite and bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
        elementwise=elementwise and bool((difference.abs() <= bounds['logit_atol']+bounds['logit_rtol']*b.abs()).all())
        exact=exact and torch.equal(a,b)
        maximum=max(maximum,float(difference.abs().max()))
        squared+=float(difference.double().square().sum())
        ref_squared+=float(b.double().square().sum())
    relative=(squared/max(ref_squared,1e-30))**.5
    return {'finite':finite,'elementwise':elementwise,'exact':exact,'max_abs':maximum,'relative_l2':relative,
            'pass':finite and elementwise and relative<=bounds['logit_relative_l2']}


def loss_sum(logits, ids):
    result=0.
    x=logits[0,:-1];y=ids[0,1:]
    for position in range(0,len(y),64):
        result+=float(torch.nn.functional.cross_entropy(x[position:position+64].float(),y[position:position+64],reduction='sum'))
    return result,len(y)
