"""Direct CUDA h/z-control checks before full-model qualification; no timings."""
import itertools
import torch
from io_utils import RUN, module, write, read
from site_controls import base_joint, extension


def main():
    torch.manual_seed(2801)
    cfg = read(RUN / 'config.json')
    if torch.cuda.get_device_name() != cfg['gpu']:
        raise RuntimeError('Approved RTX5090 required')
    bounds = cfg['numerical_bounds']
    rows = []
    diagnostics = module('run037_gpu_counter_reference', RUN / 'diagnostics.py')
    with torch.inference_mode():
        wh = torch.randn(128, 512, device='cuda', dtype=torch.bfloat16) * .02
        wz = torch.randn(128, 128, device='cuda', dtype=torch.bfloat16) * .02
        bh = torch.randn(128, device='cuda', dtype=torch.bfloat16) * .01
        bz = torch.randn(128, device='cuda', dtype=torch.bfloat16) * .01
        residual = torch.randn(16, 128, device='cuda', dtype=torch.bfloat16)
        for pattern, gated in itertools.product(['zero', 'short', 'mixed', 'dense', 'unsafe-small'], [False, True]):
            h = torch.zeros(16, 512, device='cuda', dtype=torch.bfloat16)
            z = torch.zeros(16, 128, device='cuda', dtype=torch.bfloat16)
            for x in [h, z]:
                if pattern in ['short', 'mixed', 'unsafe-small']:
                    x[:, 0] = .75; x[:, 17] = -.5
                if pattern == 'mixed': x[8:, 32:48] = .625
                if pattern == 'dense': x.normal_(0, .25)
                if pattern == 'unsafe-small': x[:, 0] = 2. ** -60
                if gated: x[:, 2] = .125; x[:, 3] = .0625  # equality survives, smaller is zero

            def evaluate(ext, skip):
                out = torch.empty_like(residual)
                stats = torch.empty((2, 8, 6), device='cuda', dtype=torch.int64)
                ext.forward(h, z, wh, wz, wh.t().contiguous(), wz.t().contiguous(), bh, bz,
                            residual, out, stats, .125, .125, gated, gated, skip, True, True)
                return out.clone(), stats.sum((0, 1)).cpu().tolist()

            reference, _ = evaluate(base_joint.extension(), False)
            for sh, sz in itertools.product([False, True], repeat=2):
                actual, stats = evaluate(extension('joint', sh, sz), True)
                delta = actual.float() - reference.float()
                relative = float(delta.norm() / reference.float().norm().clamp_min(1e-30))
                assert torch.isfinite(actual).all()
                assert torch.allclose(actual, reference, atol=bounds['logit_atol'], rtol=bounds['logit_rtol'])
                assert relative <= bounds['logit_relative_l2']
                hc = diagnostics.hybrid_counts(h.masked_fill(h < .125, 0) if gated else h, True, sh)
                zc = diagnostics.hybrid_counts(z.masked_fill(z < .125, 0) if gated else z, True, sz)
                assert stats == [hc[0], hc[1], zc[0], zc[1], hc[2], zc[2]]
                if sh == sz:
                    frozen, frozen_stats = evaluate(base_joint.extension(), sh)
                    assert torch.equal(actual, frozen) and stats == frozen_stats
                rows.append({'pattern': pattern, 'thresholding_active': gated, 'skip_h': sh, 'skip_z': sz,
                             'relative_l2': relative, 'counters': stats})
    write(RUN / 'artifacts/cuda-controls.json', {'status': 'passed', 'cases': rows,
          'scope': '40 direct h/z CUDA cases; Includes threshold boundary cases; full-model smoke also checks identical outputs and zero masks'})
    print('Passed 40 direct h/z output and counter checks', flush=True)


if __name__ == '__main__': main()
