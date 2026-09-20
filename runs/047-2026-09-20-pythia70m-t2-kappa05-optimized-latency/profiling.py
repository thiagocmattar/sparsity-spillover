"""Separate profiling traces; never use profiled samples as reported latency."""
import json
import torch
from io_utils import write
import replay


def collect(models, runners, inputs, dest):
    summary = []
    for mode in ('native', 'candidate'):
        for execution in ('graph', 'eager'):
            runner = runners[mode + '_graph']
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA],
                                        record_shapes=True) as profiler:
                for i, ids in enumerate(inputs):
                    runner.stage(ids)
                    torch.cuda.synchronize()
                    with torch.profiler.record_function(f'{mode}-{execution}-input-{i}'):
                        if execution == 'graph': runner()
                        else: replay.scaffold.forward(models[mode], ids)
                    torch.cuda.synchronize()
            path = dest / f'profile-{mode}-{execution}.json'
            profiler.export_chrome_trace(str(path))
            events = json.loads(path.read_text())['traceEvents']
            kernels = {}
            for event in events:
                if event.get('cat') != 'kernel': continue
                item = kernels.setdefault(event['name'], {'calls': 0, 'duration_us': 0.})
                item['calls'] += 1
                item['duration_us'] += event.get('dur', 0.)
            summary.append({'mode': mode, 'execution': execution, 'inputs': len(inputs),
                            'kernels': kernels, 'trace': path.name})
    write(dest / 'profile-summary.json', {'status': 'complete', 'profiles': summary,
          'interpretation': 'Instrumented structural diagnosis only; full timings are in timing.json. '
                            'Eager traces retain CPU operator shapes/correlations; graph traces retain GPU kernels.'})
