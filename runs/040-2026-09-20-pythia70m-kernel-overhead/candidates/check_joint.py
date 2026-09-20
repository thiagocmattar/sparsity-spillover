"""Synthetic gate, output, and instruction-counter checks for the new N tiles."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import traceback
from types import SimpleNamespace
import torch
import replay
from io_utils import RUN, module, read, record, verify, write
from frozen_diagnostics import hybrid_counts


def main(identifier):
    folder = RUN/'candidates'/identifier
    manifest = read(folder/'manifest.json')
    for item in manifest['files']: verify(item)
    candidate = module('check_'+identifier, folder/'joint.py')
    frozen = module('check_frozen_joint', RUN/'kernel/joint/candidate.py')
    result = {'status': 'running', 'candidate': manifest, 'checker': record(Path(__file__)), 'checks': []}
    torch.manual_seed(2801)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    try:
        with torch.inference_mode():
            w2 = torch.nn.Linear(2048,512,device='cuda',dtype=torch.bfloat16)
            wo = torch.nn.Linear(512,512,device='cuda',dtype=torch.bfloat16)
            for case in ('zero', 'short_high_index', 'complex_sparse', 'dense', 'mixed_rows', 'gate_boundary'):
                for gh, gz in ((False,False),(True,False),(False,True),(True,True)):
                    h = torch.zeros(32,2048,device='cuda',dtype=torch.bfloat16)
                    z = torch.zeros(32,512,device='cuda',dtype=torch.bfloat16)
                    if case in ('short_high_index','mixed_rows'):
                        h[:,[1025,2047]]=1; z[:,[400,511]]=1
                    if case == 'complex_sparse':
                        h[:,[0,1025,2047]]=1; z[:,[0,400,511]]=1
                    if case == 'dense': h.normal_(); z.normal_()
                    if case == 'mixed_rows': h[::3].normal_(); z[1::3].normal_()
                    if case == 'gate_boundary':
                        values=torch.tensor([.498046875,.5,.50390625],device='cuda',dtype=torch.bfloat16)
                        h[:,[0,1025,2047]]=values; z[:,[0,400,511]]=values
                    residual=torch.randn(32,512,device='cuda',dtype=torch.bfloat16)
                    previous=SimpleNamespace(w2=w2,wo=wo,gh=gh,gz=gz,th=.5,tz=.5,skip=True)
                    op, ref = candidate.Joint(previous), frozen.Joint(previous)
                    gated_h=h.masked_fill(h<.5,0) if gh else h
                    gated_z=z.masked_fill(z<.5,0) if gz else z
                    native=(w2(gated_h)+wo(gated_z))+residual
                    for skip in (True,False):
                        op.skip=ref.skip=skip; op.count=ref.count=True
                        actual=op(h,z,residual).clone(); expected=ref(h,z,residual).clone()
                        stat=op.stats.sum(tuple(range(op.stats.ndim-1))).tolist()
                        old=ref.stats.sum(tuple(range(ref.stats.ndim-1))).tolist()
                        oracle_h=hybrid_counts(gated_h,op.fast_weights,skip)
                        oracle_z=hybrid_counts(gated_z,op.fast_weights,skip)
                        oracle=oracle_h[:2]+oracle_z[:2]+[oracle_h[2],oracle_z[2]]
                        error=(actual.float()-native.float()).abs()
                        passed=(torch.equal(actual,expected) and stat==old==oracle
                                and bool(torch.isfinite(actual).all())
                                and bool((error<=.03125+.02*native.float().abs()).all()))
                        row={'case':case,'gate_h':gh,'gate_z':gz,'skip':skip,
                             'bitwise_equal_frozen':bool(torch.equal(actual,expected)),
                             'maximum_absolute_error_native':float(error.max()),
                             'counts':stat,'oracle':oracle,'pass':passed}
                        result['checks'].append(row)
                        print(row,flush=True)
                        assert passed, row
        result['status']='passed'
    except Exception:
        result['status']='failed'; result['traceback']=traceback.format_exc()
        raise
    finally:
        write(RUN/'artifacts/development'/('operator-'+identifier+'.json'),result)


if __name__ == '__main__':
    assert sys.argv[1] in ('opt002','opt003')
    main(sys.argv[1])
