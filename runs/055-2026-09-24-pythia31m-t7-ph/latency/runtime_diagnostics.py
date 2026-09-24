"""Untimed BF16 operand moments, occupancy and independently checked h/z work."""
import torch
from sparsity_research.capture import ActivationCapture
from sparsity_research.metrics import ActivationAccumulator
from tile_oracle import counts
from parallel_work_oracle import extra_scalar_products


def collect(model, data, forward, emit, blocks=338):
    accumulator = ActivationAccumulator((0., .001, .01))
    totals, histograms = {}, {}
    originals = [layer._run026_joint for layer in model.gpt_neox.layers]
    with ActivationCapture(model, ["a", "m", "h", "z", "q_post", "k_post", "v"], torch=torch) as capture:
        class CountedJoint:
            def __init__(self, op, layer):
                self.op, self.layer = op, layer

            def __call__(self, h, z, residual):
                op = self.op
                actual = [value.masked_fill(value < threshold, 0) if gate else value
                          for value, gate, threshold in ((h,op.gh,op.th),(z,op.gz,op.tz))]
                old = op.count
                op.count = True
                try:
                    output = op(h,z,residual)
                finally:
                    op.count = old
                stat = op.stats.sum(tuple(range(op.stats.ndim-1))).cpu().tolist()
                extra = extra_scalar_products(*actual, fast_weights=op.fast_weights, skip=op.skip)
                for index, site in enumerate(("h", "z")):
                    key = f"{site}.layer_{self.layer}"
                    value = actual[index]
                    capture.activations[key] = value
                    expected = counts(value, 8, op.fast_weights, op.skip, 8)
                    expected[2] += extra[index]
                    observed = [stat[index*2], stat[index*2+1], stat[index+4]]
                    if expected != observed:
                        raise RuntimeError(f"MMA/scalar counter mismatch {key}: {observed} != {expected}")
                    totals[key] = [a+b for a,b in zip(totals.get(key,[0,0,0]), observed)]
                    flat = value.reshape(-1,value.shape[-1])
                    hist = torch.bincount((flat != 0).sum(-1), minlength=flat.shape[-1]+1).cpu().tolist()
                    histograms[key] = [a+b for a,b in zip(histograms.get(key,[0]*len(hist)),hist)]
                return output
        try:
            for i, layer in enumerate(model.gpt_neox.layers):
                layer._run026_joint = CountedJoint(originals[i], i)
            with torch.inference_mode():
                for block in range(blocks):
                    capture.clear()
                    ids = torch.tensor(data[block*2048:(block+1)*2048].copy(), device="cuda", dtype=torch.long)[None]
                    forward(model, ids)
                    accumulator.update(capture.activations, torch=torch)
                    if (block+1)%32 == 0:
                        emit("runtime_diagnostics", blocks=block+1)
        finally:
            for layer, original in zip(model.gpt_neox.layers, originals):
                layer._run026_joint = original
    return dict(blocks=blocks, input_tokens=blocks*2048, documents=500 if blocks==338 else None,
        excluded_tail_tokens=1444 if blocks==338 else None, precision="BF16 runtime operands",
        activation_rows=accumulator.rows(), pooled_by_site=accumulator.pooled_by_site(),
        h_z_work=totals, work_columns=["issued_mma", "bypassed_mma", "scalar_products_including_duplicate_prepass"],
        row_nnz_histograms=histograms, interpretation="Issued work includes padded MMA rows; distinct from canonical R_model and latency.")
