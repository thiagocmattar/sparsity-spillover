"""Audit added cancellation stress against the retained implementation; no qualification override."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import traceback
from types import SimpleNamespace
import torch
import replay
from io_utils import RUN, module, read, record, verify, write
from tile_oracle import counts
from parallel_work_oracle import extra_scalar_products


def main(identifier):
    folder = RUN/'candidates'/identifier
    manifest = read(folder/'manifest.json')
    for item in manifest['files']: verify(item)
    candidate = module('check_'+identifier, folder/'joint.py')
    spec = read(folder/'spec.json')
    rows, limit = spec['M'], spec.get('short_limit',2)
    frozen = module('check_frozen_joint', RUN/'kernel/joint/candidate.py')
    result = {'status': 'running', 'candidate': manifest, 'checker': record(Path(__file__)), 'checks': []}
    torch.manual_seed(2801)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    try:
        with torch.inference_mode():
            w2 = torch.nn.Linear(2048,512,device='cuda',dtype=torch.bfloat16)
            wo = torch.nn.Linear(512,512,device='cuda',dtype=torch.bfloat16)
            for case in ('zero', 'short_high_index', 'complex_sparse', 'dense', 'mixed_rows', 'gate_boundary', 'four_values', 'eight_values', 'sixteen_values', 'cancellation'):
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
                    if case in ('four_values','eight_values','sixteen_values','cancellation'):
                        count={'four_values':4,'eight_values':8,'sixteen_values':16,'cancellation':8}[case]
                        idx=torch.arange(count,device='cuda')*31
                        values=torch.randn(32,count,device='cuda',dtype=torch.bfloat16)
                        if case=='cancellation': values[:]=torch.tensor([1024,-1024,.5,-.5,.01,-.01,2,-2],device='cuda',dtype=torch.bfloat16)
                        h[:,idx]=values;z[:,idx]=values.flip(-1)
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
                        oracle_h=counts(gated_h,rows,op.fast_weights,skip,max_short=limit)
                        oracle_z=counts(gated_z,rows,op.fast_weights,skip,max_short=limit)
                        oracle=oracle_h[:2]+oracle_z[:2]+[oracle_h[2],oracle_z[2]]
                        extra=extra_scalar_products(gated_h,gated_z,rows=rows,limit=limit,fast_weights=op.fast_weights,skip=skip)
                        oracle[4]+=extra[0];oracle[5]+=extra[1]
                        error=(actual.float()-native.float()).abs()
                        passed=(bool(((actual.float()-expected.float()).abs()<=.03125+.02*expected.float().abs()).all()) and stat==oracle
                                and bool(torch.isfinite(actual).all())
                                and bool((error<=.03125+.02*native.float().abs()).all()))
                        row={'case':case,'gate_h':gh,'gate_z':gz,'skip':skip,
                             'bitwise_equal_frozen':bool(torch.equal(actual,expected)),
                             'maximum_absolute_error_native':float(error.max()),
                             'frozen_passes_native_bound':bool(((expected.float()-native.float()).abs()<=.03125+.02*native.float().abs()).all()),
                             'matches_frozen_bound':bool(((actual.float()-expected.float()).abs()<=.03125+.02*expected.float().abs()).all()),
                             'counts':stat,'oracle':oracle,'pass':passed}
                        result['checks'].append(row)
                        print(row,flush=True)
                        # Retain all80 results even when inherited reference rounding fails.
        result['status']='audit-complete'
        result['all_bitwise_frozen']=all(row['bitwise_equal_frozen'] for row in result['checks'])
        result['all_frozen_bounds']=all(row['matches_frozen_bound'] for row in result['checks'])
        result['all_native_bounds']=all(row['pass'] for row in result['checks'])
    except Exception:
        result['status']='failed'; result['traceback']=traceback.format_exc()
        raise
    finally:
        write(RUN/'artifacts/development'/('operator-stress-audit-'+identifier+'.json'),result)


if __name__ == '__main__':
    assert sys.argv[1].startswith('opt')
    main(sys.argv[1])
