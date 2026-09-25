"""Small synthetic CUDA checks; development hardware never qualifies target science."""
import argparse
import traceback
import torch
from io_utils import RUN, write, read
from site_controls import base_joint, extension
from controls import mask
from mechanism_reference import projection_work, same_bits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--development', action='store_true')
    args = parser.parse_args()
    torch.manual_seed(2801)
    cfg = read(RUN / 'config.json')
    gpu = torch.cuda.get_device_name()
    if gpu != cfg['gpu'] and not args.development:
        raise RuntimeError('Approved RTX5090 required for target qualification')
    dest = RUN / ('prelaunch/cuda-development.json' if args.development else 'artifacts/cuda-controls.json')
    if dest.exists():
        raise FileExistsError(f'Retain prior check: {dest}')
    report = {'status': 'running', 'gpu': gpu, 'development': args.development,
              'target_qualified': False, 'cases': []}
    bounds = cfg['numerical_bounds']
    try:
        with torch.inference_mode():
            wh = torch.randn(128,512,device='cuda',dtype=torch.bfloat16)*.02
            wz = torch.randn(128,128,device='cuda',dtype=torch.bfloat16)*.02
            bh = torch.randn(128,device='cuda',dtype=torch.bfloat16)*.01
            bz = torch.randn(128,device='cuda',dtype=torch.bfloat16)*.01
            residual = torch.randn(16,128,device='cuda',dtype=torch.bfloat16)
            patterns = ['zero','one','two','mixed-rows','mixed-groups','h-only-short',
                        'z-only-short','dense','unsafe-small','unsafe-large','gate-boundary','unsafe-weights']
            for pattern in patterns:
                h = torch.zeros(16,512,device='cuda',dtype=torch.bfloat16)
                z = torch.zeros(16,128,device='cuda',dtype=torch.bfloat16)
                for x in (h,z):
                    if pattern not in ('zero','dense'):
                        x[:,0] = .75
                    if pattern in ('two','mixed-rows','mixed-groups','unsafe-small','unsafe-large','unsafe-weights'):
                        x[:,17] = -.5
                    if pattern == 'mixed-rows':
                        x[::8,32:35] = .625
                    if pattern == 'mixed-groups':
                        x[8:,32:48] = .625
                    if pattern == 'dense': x.normal_(0,.25)
                    if pattern == 'unsafe-small': x[:,0] = 2.**-60
                    if pattern == 'unsafe-large': x[:,0] = 2.**60
                    if pattern == 'gate-boundary':
                        x[:,0:4] = torch.tensor([.25,.5,.75,-.5],device='cuda',dtype=torch.bfloat16)
                if pattern == 'h-only-short': z[:,32:48] = .625
                if pattern == 'z-only-short': h[:,32:48] = .625
                gate, threshold = pattern == 'gate-boundary', .5 if pattern == 'gate-boundary' else 0.
                weight_h = wh.clone()
                if pattern == 'unsafe-weights': weight_h[0,0] = 2.**-60
                fast = pattern != 'unsafe-weights'
                trans_h, trans_z = weight_h.t().contiguous(), wz.t().contiguous()

                def evaluate(ext, skip, count=True):
                    out = torch.empty_like(residual)
                    stats = torch.full((2,8,6), -777, device='cuda',dtype=torch.int64)
                    ext.forward(h,z,weight_h,wz,trans_h,trans_z,bh,bz,residual,out,stats,
                                threshold,threshold,gate,gate,skip,fast,count)
                    torch.cuda.synchronize()
                    if not count:
                        assert bool((stats == -777).all()), 'Untimed counters leaked into count-off path'
                    return out.clone(), stats.sum((0,1)).cpu().tolist()

                frozen, _ = evaluate(base_joint.extension(), True)
                dense, _ = evaluate(base_joint.extension(), False)
                for mode in ('t00','t10','t01','t11'):
                    switches = mask(mode)
                    ext = extension(**switches)
                    actual, stats = evaluate(ext, True)
                    uncounted, _ = evaluate(ext, True, False)
                    assert same_bits(actual,uncounted), 'Counting changes outputs'
                    operands = [x.masked_fill(x < threshold,0) if gate else x for x in (h,z)]
                    hc,zc = [projection_work(x,fast_weights=fast,**switches)['legacy_counters'] for x in operands]
                    expected = [hc[0],hc[1],zc[0],zc[1],hc[2],zc[2]]
                    assert stats == expected, (pattern,mode,stats,expected)
                    relative = float((actual.float()-frozen.float()).norm()/frozen.float().norm().clamp_min(1e-30))
                    assert torch.isfinite(actual).all()
                    assert torch.allclose(actual,frozen,atol=bounds['logit_atol'],rtol=bounds['logit_rtol'])
                    assert relative <= bounds['logit_relative_l2']
                    if mode == 't11': assert same_bits(actual,frozen)
                    if mode == 't00': assert same_bits(actual,dense)
                    report['cases'].append({'pattern':pattern,'mode':mode,'counters':stats,
                        'frozen_bitwise':same_bits(actual,frozen),'relative_l2':relative})
            report.update(status='passed',target_qualified=gpu == cfg['gpu'] and not args.development,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except Exception:
        report.update(status='failed',error=traceback.format_exc())
        raise
    finally:
        write(dest,report)
    print(f'Passed {len(report["cases"])} synthetic CUDA cases on {gpu}; target_qualified={report["target_qualified"]}',flush=True)


if __name__ == '__main__': main()
