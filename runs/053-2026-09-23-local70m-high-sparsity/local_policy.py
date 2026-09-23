"""Frozen local policy; unchanged gates, BF16 branches and residual ordering."""
import torch
import triton
from local_support import RUN, load
from wide_sparse import Linear as WideLinear

def install(model, mode, base_install):
    from io_utils import module
    base_install.install(model, 'native_hz')
    module('support', RUN/'deps/run051/support.py')
    old = module('run053_model_primitives', RUN/'deps/run051/primitives.py')
    old_specs = {s['id']:s for s in old.candidates(include_ablations=True)}
    wide_specs = {s['id']:s for s in load(RUN/'wider-output-config.json')['candidates']}
    policies = load(RUN/'provenance/local-policy.json')
    policy = policies['candidate' if mode == 'candidate_no_skip' else mode]
    no_skip = mode == 'candidate_no_skip'

    def operator(linear, threshold, name):
        if name in wide_specs:
            return WideLinear(linear, threshold, wide_specs[name], no_skip=no_skip)
        spec = old_specs[name]
        if no_skip and spec['family'] == 'c':
            spec = {**spec, 'no_skip':True}
        assert spec['family'] in ('dense', 'c')
        return old.Linear(linear, threshold, spec)

    class Joint:
        native = True
        def __init__(self, previous, index):
            for attr in ('w2','wo','gh','gz','th','tz'):
                setattr(self, attr, getattr(previous, attr))
            self.h = operator(self.w2, self.th, policy[f'h.{index}'])
            self.z = operator(self.wo, self.tz, policy[f'z.{index}'])
            self.out = torch.empty((1,2048,512), device=self.w2.weight.device, dtype=torch.bfloat16)
        def __call__(self, h, z, residual):
            yh = self.h(h.reshape(2048,-1).contiguous())
            yz = self.z(z.reshape(2048,-1).contiguous())
            old.combine[(triton.cdiv(self.out.numel(),1024),)](yh,yz,residual,self.out,self.out.numel(),1024)
            return self.out

    choices = []
    for index, layer in enumerate(model.gpt_neox.layers):
        previous = layer._run026_joint
        if not previous.gh and not previous.gz:
            choices.append({'layer':index, 'h':'native', 'z':'native', 'reason':'Base without gates'})
            continue
        assert previous.gh and previous.gz
        layer._run026_joint = Joint(previous, index)
        choices.append({'layer':index,'h':policy[f'h.{index}'],'z':policy[f'z.{index}'],'no_skip':no_skip})
    return choices
