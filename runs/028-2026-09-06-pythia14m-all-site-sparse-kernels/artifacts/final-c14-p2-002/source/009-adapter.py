"""Compatibility-only K019/K018 port; fixed tiles, optional gates, skip ablation."""
from functools import lru_cache
from types import MethodType
import os
import torch
from run027_common import RUN, R26, module, sha256
from sparsity_research.pythia import expose_attention_sites, topology_metadata
from sparsity_research.sites import FixedOneSidedThreshold, FixedSymmetricThreshold

rope = module('run027_frozen_rope', R26/'autoresearch/candidates/k019/candidate.py')
joint = module('run027_frozen_joint', R26/'autoresearch/candidates/k018/candidate.py')


@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load
    major,minor=torch.cuda.get_device_capability()
    os.environ['TORCH_CUDA_ARCH_LIST']=f'{major}.{minor}'
    os.environ.setdefault('MAX_JOBS','2')
    return load(name='run027_'+sha256(RUN/'kernel.cu')[:12],sources=[str(RUN/'kernel.cu')],
                extra_cuda_cflags=['-O3','-lineinfo'],verbose=True)


class RopeGate(rope.RopeGate):
    def __init__(self, attention, backend=None):
        self.heads=attention.config.num_attention_heads
        self.dimension=attention.head_size
        self.thresholds=[]
        for site in ['q_post','k_post','v']:
            gate=getattr(attention,f'{site}_gate',None)
            if gate is not None and not isinstance(gate,FixedSymmetricThreshold):
                raise ValueError('Only absent/symmetric QKV gates are in this cohort')
            self.thresholds.append(0. if gate is None else gate.kappa)
        self.backend=backend
        self.buffers=None


class Joint(joint.Joint):
    def __init__(self,layer,skip=True,backend=None):
        self.w2,self.wo=layer.mlp.dense_4h_to_h,layer.attention.dense
        self.gh=isinstance(layer.mlp.act,(FixedOneSidedThreshold,torch.nn.ReLU))
        self.gz=isinstance(getattr(layer.attention,'z_gate',None),FixedOneSidedThreshold)
        self.th=getattr(layer.mlp.act,'kappa',0.)
        self.tz=getattr(getattr(layer.attention,'z_gate',None),'kappa',0.)
        self.skip=skip
        self.test_backend=backend
        self.backend=self.run
        self.wh=self.wz=self.out=None

    def run(self,h,z,wh,wz,bh,bz,residual,out,th,tz):
        (self.test_backend or extension().forward)(h,z,wh,wz,bh,bz,residual,out,th,tz,self.gh,self.gz,self.skip)


def install(model,mode,*,rope_backend=None,joint_backend=None):
    if mode not in {'fusion_dense','sparse','no_skip'}: raise ValueError(mode)
    metadata=topology_metadata(model)
    if metadata['topology_id'] not in {'A0','A1-H','A4-Z','A7-Z-POST'}:
        raise ValueError('Topology outside this characterization')
    expose_attention_sites(model,torch=torch)
    for layer in model.gpt_neox.layers:
        if not layer.use_parallel_residual: raise ValueError('Parallel residual required')
        attention=layer.attention
        attention._run026_rope=RopeGate(attention,rope_backend)
        attention.forward=MethodType(rope.attention_forward,attention)
        if mode!='fusion_dense':
            work=Joint(layer,skip=mode=='sparse',backend=joint_backend)
            layer._run026_joint=work
            layer.mlp.dense_4h_to_h=torch.nn.Identity()
            attention.dense=torch.nn.Identity()
            if work.gh: layer.mlp.act.forward=MethodType(joint.identity,layer.mlp.act)
            if work.gz: attention.z_gate.forward=MethodType(joint.identity,attention.z_gate)
            layer.forward=MethodType(joint.layer_forward,layer)
    return metadata
