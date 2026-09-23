"""Full-validation activation inventory for the frozen candidate's real path."""
import time
import torch
from local_support import write
from wide_sparse import Linear, expected_work

def collect(model, data, replay, dest, progress, qualified_loss):
    from sparsity_research.metrics import ActivationAccumulator, weight_statistics
    accumulator = ActivationAccumulator((0., .001, .01))
    histograms = {}; work = []; current = [0]; started = time.monotonic()
    write(dest/'candidate-weight-statistics.json', weight_statistics(model))
    class Observe:
        def __init__(self, core, index): self.core, self.index = core, index
        def __call__(self, h, z, residual):
            operands = {}
            for site, value in [('h',h),('z',z)]:
                name = f'{site}.{self.index}'
                threshold = float(torch.tensor(getattr(self.core,'t'+site), dtype=torch.bfloat16))
                raw = value.reshape(2048,-1).contiguous()
                gated = raw.masked_fill(raw < threshold, 0)
                accumulator.update({f'{site}.layer_{self.index}':gated},torch=torch)
                active = gated != 0
                info = histograms.setdefault(name,{})
                for group in (1,8,16,32,64):
                    hist = torch.bincount(active.reshape(2048//group,group,-1).any(1).sum(-1),minlength=raw.shape[-1]+1)
                    if group not in info: info[group] = hist
                    else: info[group] += hist
                op = getattr(self.core, site)
                if isinstance(op, Linear):
                    op.count = True
                    operands[name] = (raw,threshold,op)
            result = self.core(h,z,residual)
            for name,(raw,threshold,op) in operands.items():
                expected = expected_work(raw,threshold,op.spec,op.no_skip)
                assert torch.equal(op.work,expected),(current[0],name)
                work.append({'block':current[0],'site':name,'executed_reduction_tiles':int(op.work.sum()),
                             'potential_reduction_tiles':op.work.numel()*(op.k//op.spec['bk'])})
                op.count = False
            return result
    for index,layer in enumerate(model.gpt_neox.layers):
        assert layer._run026_joint.gh and layer._run026_joint.gz
        layer._run026_joint = Observe(layer._run026_joint,index)
    for block in range(338):
        current[0] = block
        ids = torch.tensor(data[block*2048:(block+1)*2048].copy(),device='cuda',dtype=torch.long)[None]
        replay.scaffold.forward(model,ids)
        if (block+1)%32 == 0 or block == 337:
            elapsed = time.monotonic()-started
            progress('candidate_diagnostics',blocks=block+1,target_blocks=338,
                     loss=qualified_loss,blocks_per_second=(block+1)/elapsed,
                     remaining_seconds=(337-block)*elapsed/(block+1))
    write(dest/'candidate-activation-statistics.json',{
        'coverage':{'documents':500,'blocks':338,'input_tokens':692224,'excluded_tail_tokens':1444},
        'path':'Candidate eager path; count-enabled kernels previously checked for bitwise agreement. Loss is from the separate numerical qualification.',
        'per_site_layer':accumulator.rows(),'pooled_by_site':accumulator.pooled_by_site(),
        'structure':{site:{f'union{gm}_nnz_hist':value.cpu().tolist() for gm,value in entries.items()} for site,entries in histograms.items()}})
    write(dest/'candidate-work.json',{'interpretation':'Executed software reduction tiles for the eight new union sites, independently checked on every validation input. Not DRAM bytes or hardware instruction counts; older union sites are covered by occupancy histograms.', 'rows':work})
