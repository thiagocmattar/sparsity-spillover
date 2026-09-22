# First assessment: efficient 70M dense and sparse execution

The question is whether kernel changes can remove the dense overhead and make
retained, lower-loss T2/Ph checkpoints faster than the 70M native PyTorch Base,
while also exploiting sparsity in T7/Pall. The evidence supports further targeted
optimization. It does not yet establish that the requested quality/speed trade-off
is achievable. The dominant problem is identified more precisely than the phrase
"the 14M kernel did not transfer": the selected 70M implementation still has an
expensive h/z matrix fallback, and it was selected for an extremely sparse,
higher-loss checkpoint.

**Measured position.** These are matched Run045 geometric-mean host latencies
on one RTX5090: BF16, batch one, 2,048 tokens, full 50,304 logits, CUDA graphs.
Quality is ordinary final-checkpoint FP16 validation; it is not the separate
BF16 kernel-qualification loss. Speedup uses the fixed native Base, 1.611048 ms.

| Checkpoint | Canonical loss | Delta from Base | Execution ms | Speedup / native Base |
|---|---:|---:|---:|---:|
| Base, native execution | 4.099767 | 0 | 1.611048 (native) | 1.000x |
| Base, opt073 | 4.099767 | 0 | 2.732112 | 0.590x |
| T2/Ph, kappa=.05 | 4.192501 | +0.092734 | 2.137268 | 0.754x |
| T2/Ph, kappa=.1 | 4.182317 | +0.082550 | 1.955582 | 0.824x |
| T7/Pall, kappa=.05 | 4.963642 | +0.863875 | 2.717419 | 0.593x |
| T7/Pall, kappa=.1 | 4.950103 | +0.850336 | 2.678416 | 0.601x |
| T7/Pall, kappa=.5 | 5.215976 | +1.116209 | 1.275788 | 1.263x |
| T7/Ph, kappa=.5 (previous search target) | 5.337238 | +1.237470 | 1.088146 | 1.481x |

Caption: the original port's Base was 3.325458 ms. Re-optimization reduced it
to 2.732112 ms, but the remaining overhead is 69.59% relative to native Base.
All 78 processes qualify; three processes per checkpoint each time 64 sequences
over seven passes. Full numerical qualification covers all 338 complete blocks
from 500 validation documents: 692,224 input tokens, 691,886 prediction tokens,
1,444 excluded tail tokens. Six of the 26 optimized checkpoints beat native Base.

Run047 later measured T2/Ph kappa=.5 at 1.214368 ms, loss 4.838034, and 1.364x
its own session's 1.656129 ms native Base. That high-threshold success does not
resolve the lower-loss gap. Its absolute timing must not use Run045's denominator.
The current training-results TeX describes these facts consistently. Older
Analysis028 text saying the endpoint's optimized latency is unmeasured is its
historical state, superseded by Analysis030/Run047.

**Dense efficiency is partly solved, but has not become the default execution
policy.** Run040's native-h/z replacement removed 1.661 ms from the original
Base path, nearly the entire 1.674 ms deficit. Run042 subsequently measured
Base with native h/z and the other optimized components at **1.297441 ms versus
1.701483 ms native Base in that session**, a 23.75% reduction. This is an actual
qualified dense control, not an estimate or sparsity effect. It supplies a
concrete starting implementation and a stronger comparator for sparse work.

The ordinary opt073 installer still puts every layer through the custom joint
h/z path. It does not route dense or moderately sparse operands to that native
replacement. The replacement helper already preserves thresholds, projection
biases and the two BF16 residual-addition roundings. Consequently, the next task
need not rediscover all dense optimizations. It needs to qualify a coherent
execution policy and determine when sparse execution improves on its efficient
dense counterpart. Run042's 1.297 ms cannot be assigned to the low-loss T2
checkpoints: their matched native-h/z replacement timings remain missing.

**The selected fast path matches the old search target.** Run042's freeze file
explicitly selects by T7/Ph kappa=.5 development latency; dense Base only had to
qualify numerically, with its latency reported. There was no dense regression
constraint or low-loss T2 objective. Run045 later assessed the frozen winner
across the grid; it did not optimize for that grid. The inherited criterion
therefore differs materially from the user's current target.

opt073 has a parallel prepass for rows with at most eight safe nonzero entries,
then an M8/N256/K16 fallback. All eight paired h/z rows must be short for a whole
fallback group to return immediately. Otherwise the fallback reinspects the
rows, computes scalar outputs for short branches and uses padded 16-row MMA
instructions for the remaining work. Short-row work can be repeated in mixed
groups. The source loads fallback weights directly and ORs active K-tile masks
across the group. Thus isolated zeros do not remove an instruction whose other
rows still require that tile. Padding, repeated inspection, weight access and
group coupling are visible mechanisms; their individual latency costs have not
been isolated.

| Checkpoint | h/z exact zeros (%) | h/z mean nonzeros per row | h/z rows with <=8 nonzeros (%) | h/z issued MMA fraction (%) |
|---|---:|---:|---:|---:|
| T2/Ph .05 | 98.32 / 93.32 | 34.35 / 34.20 | 32.54 / 0.19 | 52.93 / 73.11 |
| T2/Ph .1 | 98.92 / 96.85 | 22.14 / 16.15 | 45.09 / 19.75 | 41.87 / 52.74 |
| T7/Pall .05 | 89.47 / 80.93 | 215.59 / 97.63 | 0.00 / 0.08 | 99.41 / 93.22 |
| T7/Pall .1 | 92.26 / 88.07 | 158.50 / 61.07 | 0.00 / 1.54 | 98.24 / 82.78 |
| T7/Pall .5 | 99.86 / 99.84 | 2.78 / 0.80 | 92.71 / 99.34 | 3.74 / 0.84 |
| T7/Ph .5 | 99.98 / 99.97 | 0.45 / 0.16 | 99.98 / 100.00 | 0.019 / 0.00039 |

Caption: each site pools 4,153,344 BF16 rows from the full 338-block diagnostic.
The h row has 2,048 features; z has 512. Counts are summed before division.
Rounded 100% does not mean every row. <=8 occupancy is necessary, not sufficient,
for the safe scalar path; marginal row histograms do not identify joint h/z
eight-row group completion. MMA fractions use the actual kernel's padded
potential-instruction denominator and exclude separately counted scalar work.
They are neither R_model nor runtime savings.

T2/Ph .1 already reaches 15.212% canonical R_model against its 15.443%
topology ceiling. That proximity describes logical products, not how well the
kernel exploits the pattern. The dense vocabulary head alone accounts for
50.576% of the full-sequence logical-product denominator and remains dense.
Neither percentage is a latency fraction or a bound on attainable speedup.

At T2/Ph .1, h is especially heterogeneous: layers 0 and 5 have mean row
occupancies 64.14 and 39.01, while layers 1 and 2 have 3.02 and 3.77. A uniform
h/z policy across all layers leaves a concrete opportunity for a better
execution rule. The full group correlations and their timing consequences
still need measurement; they cannot be recovered from marginal histograms alone.

**Profiles already localize the low-loss bottleneck.** Run045 retained profiles
for Base, T7/Ph .5 and T2/Ph .1. The following are summed GPU kernel durations
per input over four separately instrumented inputs, not benchmark host latency.

| Component | Base (ms) | T2/Ph .1 (ms) | T7/Ph .5 (ms) |
|---|---:|---:|---:|
| h/z fallback | 1.683 | 0.899 | 0.035 |
| h/z prepass | 0.021 | 0.048 | 0.056 |
| Dense vocabulary head | 0.445 | 0.442 | 0.435 |
| Native a/m projections | 0.260 | 0.259 | 0.258 |
| Dense attention | 0.218 | 0.217 | 0.217 |

The .1 h/z path totals about 0.947 ms, versus 0.091 ms at the former search
target. Its 0.899 ms fallback alone exceeds the 0.345 ms full-model saving
needed to reach native Base parity. This makes targeted work plausible, but
does not prove that the required saving is available against an efficient
replacement. Head and attention optimizations matter, yet the evidence gives
h/z fallback higher priority for this goal. At .05 the matching component
profile was not retained, although its occupancy and work counts were.

**What is still missing, in priority order.**

1. An approved quantitative objective: allowable loss increase, maximum Base
   slowdown, and minimum practically meaningful speedup. For illustration only,
   1.1x/1.2x native Base means at most 1.465/1.343 ms in Run045's session.
   The .05 checkpoint needs 24.6% lower latency merely for parity and 37.2%
   for 1.2x; .1 needs 17.6% and 31.3%. These are arithmetic targets, not
   forecasts. A +0.083 to +0.093 cross-entropy difference and one training seed
   do not establish statistical quality equivalence.
2. Matched native-h/z and separate h/z execution controls at the requested
   .05/.1 checkpoints, alongside native Base, efficient dense Base and the
   same checkpoint's native implementation. Existing Run042 controls focus on
   T7/Ph .5. Its 58.3% skip-on saving against an inefficient custom fallback
   does not measure savings over an efficient dense path. The practical native
   replacement benefit there is 17.4%, including fusion/layout effects.
3. A kernel policy for dense and moderately sparse rows. The evidence motivates
   retaining the efficient dense backbone, then testing a native fallback,
   independent h/z choices, or a faster moderately sparse matrix schedule.
   Reuse inspection summaries and avoid recomputing completed short rows where
   it helps. Packing or compaction, shared weight staging, different row groups
   and longer scalar paths are candidates to measure, not promised improvements.
   Any activation-dependent routing must include inspection, packing, launches
   and synchronization in the timed workload and work under the graph protocol.
4. Selection on the requested workload: include Base, T2/Ph .05/.1 and declared
   T7/Pall thresholds, rather than selecting solely on T7/Ph .5. Preserve
   training-only development inputs and freeze before full validation. One
   execution policy may use declared shape/site/layer and measured operand
   properties; it must not special-case checkpoint or validation-input identity.
5. Attribution and correctness at that regime. Retain joint h/z group occupancy,
   mixed-group/duplicate work, per-layer fallback time, and independent h/z
   controls. Keep the existing full-logit bounds and complete validation. The
   eight inherited extreme-cancellation stress failures are documented, despite
   all final model checks passing; new accumulation schedules need explicit
   numerical assessment. A strong compiled-dense comparator is also absent from
   this matched grid; it would strengthen a broader systems claim, but is not a
   prerequisite for the narrower declared native-PyTorch comparison.

For T7/Pall, opt073 exploits h/z only: a/m and attention use dense execution.
Its extra model-wide logical sparsity therefore does not all become executable
savings. The .05/.1 h/z operands themselves issue almost all h matrix instructions
under the existing grouping. Re-enabling attention skipping is not an established
solution: the retained 14M ablations found overhead, and 70M needs its own evidence.
Further, none of the retained T7/Pall checkpoints has loss close to Base; kernel
optimization preserves each checkpoint's quality and cannot remove its training
quality cost. This does not prevent testing runtime improvement across T7/Pall.

The lean next design would first measure the missing practical controls at the
existing low-loss checkpoints, then optimize only the components that those
controls justify. The checkpoints, full validation cache, training development
inputs, numerical harness, diagnostics and raw evidence already exist. More
training is not required to answer the immediate implementation question.
Intermediate kappas could address a later quality frontier question, but they
are not the first missing ingredient for speeding up .05/.1.

**Repository readiness and limits.** Runs040/042 contain the diagnosis, immutable
candidate history and replacement controls; Run045 provides the matched grid;
Run047/Analysis030 adds the later endpoint; Analysis034 supplies the current
figure coordinates. Analysis032 provides a lean extracted kernel companion,
but its relocated code has not been requalified on a GPU. Use frozen measured
sources as the reference and requalify any extraction used in a future run.
T2 means operational HZ=(h,z), not the historical A2=(m,h) registry entry.
The top-level research index had stale next-run/analysis counters (048/031),
although Run048 and Analyses031--034 already existed; this assessment updates
the counters and adds its evidence pointer. Historical runs remain unchanged.

The CPU audit checked 78 retained qualified process records, recomputed all
26 matched timing triples from raw samples, and reconciled all 26 occupancy
diagnostics with integer zero counts. It records 321 source hashes. It is not
a new GPU benchmark, a claim of generalization to decode/training/batching, or
an approval to launch. Measurements remain one hardware type, one full-sequence
workload, one training seed and separate documented timing sessions.

Sources and generating script:

- [01_assess.py](../01_assess.py) and [assessment.json](../data/assessment.json).
- [Run040 diagnosis](../../../runs/040-2026-09-20-pythia70m-kernel-overhead/observations/001-overhead-and-optimization.md).
- [Run042 selection](../../../runs/042-2026-09-20-pythia70m-sparse-scale/provenance/final-selection.json)
  and [joint CUDA implementation](../../../runs/042-2026-09-20-pythia70m-sparse-scale/candidates/opt063/joint.cu).
- [Run045 benchmark](../../../runs/045-2026-09-20-pythia70m-kernel-grid/02_benchmark.py),
  [controls](../../../runs/045-2026-09-20-pythia70m-kernel-grid/controls.py),
  [matched results](../../../runs/045-2026-09-20-pythia70m-kernel-grid/results/matched-grid.json).
- [Separate-session controls](../../028-2026-09-20-70m-optimized-grid/data/retained-controls.json).
- [Later T2 endpoint](../../030-2026-09-20-70m-t2-optimized-endpoint/observations/001-later-70m-endpoint.md).
- [Current coordinates](../../034-2026-09-21-70m-latency-quality-frontier/data/figure-data.json)
  and [manuscript](../../../manuscript/draft/training-results.tex).
