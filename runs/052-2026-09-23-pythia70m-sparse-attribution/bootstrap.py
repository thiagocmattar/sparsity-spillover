"""Explicit bindings to the immutable Run049 measurement implementation."""
import importlib.util
import os
import sys
from support import BASE, RUN, read, sha

sys.path.insert(0,str(BASE))
import io_utils as base_io
import replay
sys.path.insert(0,str(RUN))

def bind(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod
    spec.loader.exec_module(mod)
    return mod

controls=bind('controls',BASE/'controls.py')
base_install=bind('run052_base_install',BASE/'install.py')

def setup():
    import torch, numpy as np, transformers, platform
    cfg=read(BASE/'config.json')
    got={'python':'.'.join(platform.python_version().split('.')[:2]),
         'torch':torch.__version__.split('+')[0], 'transformers':transformers.__version__,
         'numpy':np.__version__, 'cuda':torch.version.cuda}
    assert got==cfg['runtime'],got
    assert torch.cuda.get_device_name()==cfg['gpu']
    assert not os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    torch.manual_seed(cfg['runtime_seed'])
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    return cfg

def checkpoint(cid):
    manifest = RUN/'provenance/reference14.json' if cid.startswith('d') else BASE/'provenance/inputs.json'
    return next(c for c in read(manifest)['checkpoints'] if c['id']==cid)

def model(cid, custom=True):
    import torch, transformers
    from sparsity_research.pythia import load_checkpoint_pythia
    row=checkpoint(cid)
    root = RUN if cid.startswith('d') else BASE
    for item in row['files']+row['provenance']: base_io.verify(item, root)
    m=load_checkpoint_pythia(transformers.AutoModelForCausalLM,root/row['checkpoint'],torch=torch)
    m=m.to('cuda',dtype=torch.bfloat16).eval()
    m.set_attn_implementation('sdpa');m.config.use_cache=False
    if custom: install_backbone(m)
    return m

def verify_baseline():
    # Run049's verified archive cache is a symlink to local /tmp storage.
    # Verify bytes and sizes in that declared cache, not record()'s no-symlink
    # packaging predicate. The persistent original is retained by Run049.
    rows=[r['copy'] for r in read(BASE/'provenance/reuse.json')['files']]
    rows+=read(BASE/'provenance/source-freeze.json')['files']
    for row in rows:
        path=BASE/row['path']
        assert '..' not in path.relative_to(BASE).parts
        assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],row['path']


def install_backbone(model):
    if model.config.hidden_size == 512:
        return base_install.install(model, 'native_hz')
    compatibility = bind('run052_14m_compat', RUN/'hz_adapter14.py')
    compatibility.install(model, 'sparse')
    frozen = bind('run052_14m_k050', replay.R28/'candidates/k050/candidate.py')
    return frozen.install(model, shortcut=False, round_p=False, skip=True, projection_skip=True)
