"""GPU checks for capacity boundaries, gates, mixed groups and exact work counters."""
from types import SimpleNamespace
import traceback
import torch
import replay
from short_rows import Joint
from io_utils import RUN, module, write, record
from tile_oracle import counts
from parallel_work_oracle import extra_scalar_products


def main():
    torch.manual_seed(4901)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    frozen = module('run049_original_joint', RUN / 'candidates/opt063/joint.py')
    result = {'status': 'running', 'checks': [], 'source': record(RUN / 'short_rows.cu')}
    try:
        with torch.inference_mode():
            w2 = torch.nn.Linear(2048, 512, device='cuda', dtype=torch.bfloat16)
            wo = torch.nn.Linear(512, 512, device='cuda', dtype=torch.bfloat16)
            for limit in (8, 16, 32, 64):
                for case in ('zero', 'below_limit', 'at_limit', 'above_limit', 'mixed', 'dense', 'gate_boundary'):
                    for gated in (False, True):
                        h = torch.zeros(32, 2048, device='cuda', dtype=torch.bfloat16)
                        z = torch.zeros(32, 512, device='cuda', dtype=torch.bfloat16)
                        count = {'zero': 0, 'below_limit': limit-1, 'at_limit': limit,
                                 'above_limit': limit+1, 'mixed': limit, 'dense': 0, 'gate_boundary': 3}[case]
                        for x in (h, z):
                            indices = torch.linspace(0, x.shape[1]-1, count, device='cuda').long() if count else []
                            if count:
                                x[:, indices] = .75
                            if case == 'dense':
                                x.normal_()
                            if case == 'mixed':
                                x[::3].normal_()
                            if case == 'gate_boundary':
                                x[:, indices] = torch.tensor([.498046875, .5, .50390625], device='cuda', dtype=torch.bfloat16)
                        if case == 'mixed':
                            z[1::3].normal_()
                        residual = torch.randn(32, 512, device='cuda', dtype=torch.bfloat16)
                        old = SimpleNamespace(w2=w2, wo=wo, gh=gated, gz=gated, th=.5, tz=.5, skip=True)
                        op = Joint(old, limit)
                        gh = h.masked_fill(h < .5, 0) if gated else h
                        gz = z.masked_fill(z < .5, 0) if gated else z
                        native = (w2(gh) + wo(gz)) + residual
                        for skip in (True, False):
                            op.skip, op.count = skip, True
                            actual = op(h, z, residual).clone()
                            work = op.stats.sum((0, 1, 2)).tolist()
                            expected_h = counts(gh, 8, op.fast_weights, skip, limit)
                            expected_z = counts(gz, 8, op.fast_weights, skip, limit)
                            extra = extra_scalar_products(gh, gz, limit=limit, fast_weights=op.fast_weights, skip=skip)
                            expected = expected_h[:2] + expected_z[:2] + [expected_h[2]+extra[0], expected_z[2]+extra[1]]
                            delta = actual.float() - native.float()
                            passed = work == expected and bool(torch.isfinite(actual).all()) and bool((delta.abs() <= .03125 + .02 * native.float().abs()).all())
                            relative = float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(native.float()).clamp_min(1e-12))
                            passed = passed and relative <= .02
                            if limit == 8:
                                ref = frozen.Joint(old)
                                ref.skip, ref.count = skip, True
                                previous = ref(h, z, residual).clone()
                                passed = passed and torch.equal(actual, previous) and work == ref.stats.sum((0, 1, 2)).tolist()
                            op.count = False
                            passed = passed and torch.equal(actual, op(h, z, residual))
                            row = {'limit': limit, 'case': case, 'gates': gated, 'skip': skip,
                                   'counts': work, 'oracle': expected, 'maximum_absolute_error': float(delta.abs().max()),
                                   'relative_l2': relative, 'pass': passed}
                            result['checks'].append(row)
                            print(row, flush=True)
                            if not passed:
                                raise AssertionError(row)
            result['status'] = 'passed'
    except Exception:
        result.update(status='failed', traceback=traceback.format_exc())
        raise
    finally:
        write(RUN / 'artifacts/operator-checks.json', result)


if __name__ == '__main__':
    main()
