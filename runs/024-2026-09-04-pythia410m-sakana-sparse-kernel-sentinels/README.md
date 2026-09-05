# Run 024 — Pythia-410M Sakana-derived sparse-kernel sentinels

**Status:** complete and independently verified; artifacts retrieved; RunPod
Pod deleted; no GPU compute or endpoint remains.

`config.yaml` remains the byte-identical sealed prelaunch input and therefore
retains its historical `implemented_not_launched` status field. This README,
the result verification, and the closeout record are the post-execution state.

## Question

Does increasing the Pythia workload from 70M to 410M make the verified
Sakana-derived exact-ELL linear path large enough to amortize sparse packing,
dispatch, and kernel costs? Does measured full-sequence prefill speed track the
opportunity covered by the executed kernels rather than total `R_model`?

This is an inference benchmark of completed Run-019 checkpoints. It does not
train, tune, or repair Pythia-410M. Its worse loss than Pythia-70M is retained
as a quality caveat rather than used as an exclusion criterion.

## Fixed evidence and sentinel set

The six checkpoints are A0, A1-H, A4-OL1 at `kappa={0,0.5}`, and A7-OL1 at
`kappa={0,0.5}`. They are the Run-019 seed-1234 random-pretraining
realizations after 712 optimizer boundaries and 1,493,172,224 MiniPile input
tokens. Released weights were not loaded. Only model, config, checkpoint
metadata, and canonical logical-product records are transferred; optimizer
state is excluded.

Pythia-410M has 24 layers, hidden width 1,024, FFN width 4,096, 16 heads of
width 64, vocabulary 50,304, and sequence length 2,048. The four covered
linear shapes are therefore `1024x3072`, `1024x4096`, `4096x1024`, and
`1024x1024`. Attention remains a separate Q-only/V-only per-head benchmark.
Its matrix shapes do not grow from 70M because head width and sequence length
remain 64 and 2,048; only the number of heads and layers grows.

## Matched measurement

The run reuses the frozen Run-023 Sakana-derived patch and benchmark behavior.
It preserves every signed nonzero BF16 value, performs no pruning or capacity
truncation, and leaves the LM head and full-model attention dense. A0/A1-H
replace only `h -> W2`; A4/A7 replace `a -> QKV`, `m -> W1`, `h -> W2`, and
`z -> Wo`.

Every source-loss and dense/sparse-equivalence pass covers all 338 complete
2,048-token validation blocks (692,224 evaluated input tokens from all 500
documents), reporting the excluded 1,444-token tail. Source FP32 parameters
under FP16 autocast use batch four. Dense/sparse BF16 equivalence and canonical
attention opportunity use batch one.

Batch one is the primary runtime endpoint and is exactly comparable to Run
023's batch-one result. Batch 32 is conditional on a representative remote
memory calibration retaining at least 10% device-memory headroom; it is not
required for the primary answer. Timing uses five warmups, seven randomized
paired blocks, and the same primitive and full-model target durations as Run
023.

## Correctness and interpretation

The official SparseLM0.5B benchmark remains the environment positive control.
The run stops on source/blob/hash mismatch, failed signed packing, unsafe ELL
capacity, primitive relative-L2 error above 0.02, source-loss mismatch above
0.0002, or sparse/dense complete-validation loss mismatch above 0.001. A
sparse slowdown is a valid result.

Robust full-model break-even requires both paired median and paired p10
speedup above one. Primitive kernel-only, pack-plus-kernel, occupancy, exact
and near-zero counts, RMS/L2, peak memory, canonical `R_model`, linear
`R_covered`, and separately extended A7 attention coverage are retained.
`R_model` remains a logical-product opportunity, not a runtime estimate.

The worse 410M loss limits any quality-speed Pareto claim. The benchmark can
show whether these actual fixed-token checkpoints amortize systems overhead;
it cannot establish how a fully trained 410M model would behave. The six
selected points are topology-confounded and support only descriptive
association.

## Compute and lifecycle

One sequential Hopper Pod is used so every condition shares the same device.
H100 NVL is preferred to match Run 023. If unavailable, H200/H100 is allowed,
but the matched Run-023 sentinels must be rerun on that device before making a
cross-scale timing claim. The provisional expectation is 1.5–3 GPU-hours with
a six-hour termination backstop. Live price, stock, balance, storage, and the
maximum billable envelope are recorded immediately before provisioning.

The source payload is approximately 9.1 GiB of model-only checkpoints plus
small validation and code records. Logs and outputs remain under `/workspace`,
the worker is detached, and monitoring is read-only at ten-minute intervals.
Results are retrieved and SHA-256 verified locally before Pod termination.
No new checkpoint is produced or required. Post-hoc TEAL endpoints and the six
interior trained thresholds are deferred.

The sealed input inventory contains 71 regular files (9.063 GiB), including
only the six model checkpoints, their configs/metadata/logical-product records,
the validation cache, and the pinned harness. It contains no optimizer or
training state and no credentials. Local static preflight passed for all six
checkpoints, Python compilation passed, 12 focused Run-024 tests passed, and
the full bootstrap suite passed (219 tests). Bash is unavailable on the local
Windows host, so both shell sources are syntax-checked on Linux before worker
startup.

## Outcome

All six sentinels passed complete validation, dense/sparse equivalence, signed
packing, primitive correctness, and timing verification. The official
SparseLM0.5B positive control reached `1.2739x`. None of 432 Pythia-410M
linear primitives, 48 A7 attention compositions, or 12 full-model condition ×
batch timings broke even. The best full-model results were `0.3275x` at batch
one (A1-H) and `0.4989x` at batch 32 (A4-OL1, `kappa=0.5`).

The clean six-condition attempt reached its final post-timing invariant in
57m53s. A7-OL1 `kappa=0.5` was rerun from the start after its H100 V-only
opportunity exceeded the pinned A100 P/V union by 60,013,381 products (about
3.7 ppm). The correction retained exact H100 component counts and made the
cross-device extended canonical `R_covered` null; it did not cap counts or
change canonical `R_model`. The five prior condition files remained byte-for-
byte identical across recovery. The recovered six-condition cohort passed the
remote verifier and an isolated local verifier.

The exact same physical H100 NVL as Run 023 was used. Every 410M batch-one
sentinel is slower relative to dense than its matched 70M result. At batch 32,
only high-threshold A7 improves slightly (`0.405→0.432x`) and remains below
break-even. This refutes the narrow hypothesis that moving from 70M to 410M is
enough to amortize this exact-ELL path. See
`observations/001-pythia410m-sentinel-sparse-kernel-feasibility.md` and
Analysis 014 for the full reduction.

The Pod existed for approximately 3.2 hours across setup, retries, diagnostics,
review pauses, the final attempt, and recovery. Rate-times-duration is about
`$8.4` at `$2.59/GPU-hour` plus transient storage. The billing API had posted
only `$5.8007` at the first closeout query and was missing later buckets, so
that value is provisional. The final audit found zero Pods, zero endpoints,
and the unchanged pre-existing 100 GB volume `9luykg5yc3`.

## Paper boundary

This run tests the manuscript distinction between logical opportunity and
measured speed. A positive result can support a systems-calibration statement
about matrix scale and kernel coverage; a negative result delimits the current
kernel/workload combination. It does not revise the fixed-token 410M training
interpretation without a separate, user-approved manuscript change.
