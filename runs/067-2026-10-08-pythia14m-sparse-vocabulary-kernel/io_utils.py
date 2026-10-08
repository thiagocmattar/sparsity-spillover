"""Run-owned immutable identities and atomic evidence writes."""
import hashlib
import json
import os
from pathlib import Path

RUN = Path(__file__).resolve().parent
REPO = RUN.parents[1]
SOURCE = REPO / 'runs/066-2026-10-08-pythia14m-vocabulary-tile-gates'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n',
                         encoding='utf-8', newline='\n')
    temporary.replace(path)


def record(path, root=REPO):
    path = Path(path).resolve()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': path.relative_to(root).as_posix(), 'bytes': path.stat().st_size, 'sha256': digest}


def verify(row, root=REPO):
    root = Path(root).resolve()
    path = (root/row['path']).resolve()
    if not path.is_relative_to(root) or path.is_symlink() or record(path, root) != row:
        raise ValueError('File identity mismatch: '+row['path'])
    return path


def load_model(checkpoint):
    import sys
    sys.path[:0] = [str(SOURCE), str(REPO/'src')]
    import topology
    topology.register()
    from head_site import load_checkpoint_pythia
    import torch
    from transformers import AutoModelForCausalLM
    model = load_checkpoint_pythia(AutoModelForCausalLM, checkpoint, torch=torch)
    model = model.to('cuda', dtype=torch.float32).eval()
    model.set_attn_implementation('sdpa')
    model.config.use_cache = False
    return model


def configure_cuda(seed):
    import torch
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG'):
        raise ValueError('The matched default cuBLAS workspace is required')
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False


def source_identity(*, development):
    """Published source commit accompanies the byte-verified minimal cloud checkout."""
    commit = os.environ.get('RUN067_SOURCE_COMMIT')
    if not development and (commit is None or len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit)):
        raise ValueError('The verified deployment source Git commit is required')
    return {'git_commit':commit,'runtime_inventory':record(RUN/'provenance/upload-inventory.json'),
            'scope':'minimal checkout; runtime bytes verified against the sealed inventory; unrelated local untracked files excluded'}
