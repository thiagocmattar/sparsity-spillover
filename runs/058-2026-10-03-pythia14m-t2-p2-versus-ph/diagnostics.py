"""Final validation sign/energy and aligned occupancy, without changing gates."""
from contextlib import ExitStack
from sparsity_research.capture import ActivationCapture
from sparsity_research.evaluation import evaluate_complete_blocks
from sparsity_research.metrics import ActivationAccumulator
from frozen import load_run004_module
from optimizer_boundary import recipe_attention_context
from run_config import mapping, require_validation_coverage

_BASE = load_run004_module('_run058_diagnostics_base', 'diagnostics.py')
AttentionOutputCapture = _BASE.AttentionOutputCapture
def logical_product_validation(**kwargs):
    coverage, measured, ceiling = _BASE.logical_product_validation(**kwargs)
    ceiling['unit'] = 'logical scalar multiplication products per one complete uncached sequence across all layers and LM head'
    return coverage, measured, ceiling

def gate_modules(model):
    for i, layer in enumerate(model.gpt_neox.layers):
        for site, module in (('h',layer.mlp.act), ('z',layer.attention.z_gate)):
            yield f'{site}.layer_{i}', module

class GateInputCapture:
    """Capture h/z gate inputs and the ungated affine LN outputs at a/m."""
    def __init__(self,model):
        self.model, self.activations, self.handles = model, {}, []

    def __enter__(self):
        for name,module in gate_modules(self.model):
            def hook(_module, inputs, name=name):
                self.activations[name] = inputs[0]
            self.handles.append(module.register_forward_pre_hook(hook))
        for i,layer in enumerate(self.model.gpt_neox.layers):
            for site,module in (('a',layer.input_layernorm),('m',layer.post_attention_layernorm)):
                def output_hook(_module,_inputs,value,name=f'{site}.layer_{i}'):
                    self.activations[name] = value
                self.handles.append(module.register_forward_hook(output_hook))
        return self

    def __exit__(self,*args):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def clear(self):
        self.activations.clear()

class SignEnergyAccumulator:
    def __init__(self):
        self.rows_by_name = {}

    def update(self,name,before,after,spec,*,torch):
        native = before.detach()
        x, y = native.float(), after.detach().float()
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise RuntimeError(f'Nonfinite gate diagnostic: {name}')
        k = float(spec['kappa'])
        # Match the gate's comparison dtype: FP16 kappa=.1 is represented as
        # .0999755859375. Widening before the comparison changes equality.
        retained = native >= k if spec['operator'] == 'one_sided_threshold' else native.abs() >= k
        expected = x.masked_fill(~retained,0)
        if not torch.equal(expected,y):
            raise RuntimeError(f'Gate output differs from declared operator: {name}')
        row = self.rows_by_name.setdefault(name, {'name':name, 'operator':spec['operator'],
            'kappa':k, 'input_dtype':str(native.dtype),
            'effective_kappa':float(torch.tensor(k,dtype=native.dtype).item()),
            'elements':0, 'pre_negative':0, 'pre_zero':0, 'pre_positive':0,
            'post_negative':0,'post_zero':0,'post_positive':0,'rejected_count':0,
            'rejected_negative_count':0,'rejected_large_negative_count':0,
            'pre_square_sum':0.0,'post_square_sum':0.0,'rejected_square_sum':0.0,
            'rejected_large_negative_square_sum':0.0})
        row['elements'] += x.numel()
        for label,value in (('pre',x),('post',y)):
            for sign,mask in (('negative',value<0),('zero',value==0),('positive',value>0)):
                row[f'{label}_{sign}'] += int(mask.sum().item())
            row[f'{label}_square_sum'] += float(value.double().square().sum().item())
        rejected = ~retained
        large_negative = rejected & (x<0) & (native.abs()>=k)
        row['rejected_count'] += int(rejected.sum().item())
        row['rejected_negative_count'] += int((rejected & (x<0)).sum().item())
        row['rejected_large_negative_count'] += int(large_negative.sum().item())
        row['rejected_square_sum'] += float(x[rejected].double().square().sum().item())
        row['rejected_large_negative_square_sum'] += float(x[large_negative].double().square().sum().item())

    def rows(self):
        return [dict(self.rows_by_name[k]) for k in sorted(self.rows_by_name)]

    def pooled(self):
        return pool_additive(self.rows(), excluded=('operator','kappa','input_dtype','effective_kappa'))

def pool_additive(rows, *, excluded=()):
    groups = {}
    for row in rows:
        site = row['name'].split('.layer_')[0]
        dst = groups.setdefault(site, {'name':site})
        for key,value in row.items():
            if key in ('name',*excluded):
                continue
            dst[key] = dst.get(key,0)+value
    return [groups[k] for k in sorted(groups)]

class OccupancyAccumulator:
    """Sequence-local [token,feature] aligned tiles; pool integer counts first."""
    def __init__(self, shapes=((8,16),(16,16))):
        self.shapes = tuple(tuple(s) for s in shapes)
        self.data = {}

    def update(self,name,value,*,torch):
        x = value.detach()
        if x.ndim != 3:
            raise ValueError('Occupancy requires [batch,token,feature]')
        b,t,d = x.shape
        if any(t%tr or d%tc for tr,tc in self.shapes):
            raise ValueError('Declared tiles must completely partition each sequence')
        row = self.data.setdefault(name, {'name':name,'rows':0,
            'row_nonzero_histogram':[0]*(d+1),
            'tiles':{f'{tr}x{tc}':{'empty':0,'total':0} for tr,tc in self.shapes}})
        row['rows'] += b*t
        counts = torch.count_nonzero(x,dim=-1).reshape(-1)
        histogram = torch.bincount(counts, minlength=d+1).cpu().tolist()
        row['row_nonzero_histogram'] = [a+b for a,b in zip(row['row_nonzero_histogram'],histogram,strict=True)]
        for tr,tc in self.shapes:
            tiles = x.reshape(b,t//tr,tr,d//tc,tc).permute(0,1,3,2,4)
            empty = (tiles==0).all(dim=-1).all(dim=-1)
            target = row['tiles'][f'{tr}x{tc}']
            target['empty'] += int(empty.sum().item())
            target['total'] += empty.numel()

    def rows(self):
        return [self.data[k] for k in sorted(self.data)]

    def pooled(self):
        dst = {}
        for row in self.rows():
            site = row['name'].split('.layer_')[0]
            if site not in dst:
                dst[site] = {'name':site, 'rows':0,
                    'row_nonzero_histogram':[0]*len(row['row_nonzero_histogram']),
                    'tiles':{k:{'empty':0,'total':0} for k in row['tiles']}}
            target = dst[site]
            target['rows'] += row['rows']
            target['row_nonzero_histogram'] = [a+b for a,b in zip(target['row_nonzero_histogram'],row['row_nonzero_histogram'],strict=True)]
            for tile,counters in row['tiles'].items():
                for key,count in counters.items():
                    target['tiles'][tile][key] += count
        for row in dst.values():
            for counters in row['tiles'].values():
                counters['empty_fraction'] = counters['empty']/counters['total']
        return [dst[k] for k in sorted(dst)]

def activation_diagnostic_validation(*,model,tokens,config,torch,np):
    diagnostics = config['diagnostics']
    requested = diagnostics['activation_sites']
    gates = model.config.site_gates
    thresholds = tuple(sorted(set(float(v) for v in diagnostics['near_zero_thresholds']) | {float(g['kappa']) for g in gates.values()}))
    moments = ActivationAccumulator(thresholds)
    energy = SignEnergyAccumulator()
    occupancy = OccupancyAccumulator(diagnostics['tile_shapes'])
    device = next(model.parameters()).device
    with ExitStack() as stack:
        capture = stack.enter_context(ActivationCapture(model,[s for s in requested if s!='attention_output'],torch=torch))
        output = stack.enter_context(AttentionOutputCapture(model))
        before = stack.enter_context(GateInputCapture(model))
        def observe(_output,_n):
            combined = {**capture.activations,**output.activations}
            if len(combined) != len(model.gpt_neox.layers)*len(requested) or len(before.activations)!=len(model.gpt_neox.layers)*4:
                raise RuntimeError('Incomplete diagnostic capture')
            moments.update(combined,torch=torch)
            moments.update({name.replace('.layer_','_pre.layer_'):x for name,x in before.activations.items()
                            if name.split('.')[0] in ('a','m')},torch=torch)
            for name,x in before.activations.items():
                site = name.split('.')[0]
                if site in gates:
                    energy.update(name,x,combined[name],gates[site],torch=torch)
                occupancy.update(name,combined[name],torch=torch)
            capture.clear(); output.clear(); before.clear()
        with recipe_attention_context(torch,device):
            coverage = evaluate_complete_blocks(model=model,tokens=tokens,
                block_size=config['data']['sequence_length'],batch_size=config['validation']['batch_size'],
                device=device,torch=torch,np=np,autocast_dtype=torch.float16 if device.type=='cuda' else None,
                after_batch=observe)
    require_validation_coverage(coverage,config)
    if any(row['nonfinite'] for row in moments.rows()):
        raise RuntimeError('Nonfinite activation statistics')
    affine = [{'layer':i,'site':site,'weight':ln.weight.detach().float().cpu().tolist(),
               'bias':ln.bias.detach().float().cpu().tolist()}
              for i,layer in enumerate(model.gpt_neox.layers)
              for site,ln in (('a',layer.input_layernorm),('m',layer.post_attention_layernorm))]
    return coverage, {'rows':moments.rows(),'pooled_by_site':moments.pooled_by_site(),
        'near_zero_thresholds':list(thresholds),'sign_energy_rows':energy.rows(),
        'sign_energy_pooled_by_site':energy.pooled(), 'occupancy_rows':occupancy.rows(),
        'occupancy_pooled_by_site':occupancy.pooled(),'layernorm_affine':affine,
        'tile_unit':'nonoverlapping token x feature tiles within each sequence, no padding',
        'site_definition':{'a_pre':'ungated learned affine input_layernorm output',
            'm_pre':'ungated learned affine post_attention_layernorm output',
            'a':'ungated LayerNorm output before QKV','m':'ungated LayerNorm output before W1',
            'h':'gate replacing GELU before W2','z':'concatenated PV post z_gate before W_o',
            'attention_output':'after W_o before zero dropout/residual'}}
