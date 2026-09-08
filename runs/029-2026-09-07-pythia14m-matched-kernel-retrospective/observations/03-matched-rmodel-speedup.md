# Canonical R_model, final acceleration and sparse-path controls

## Question and method

Does the final specialized kernel translate the retained train-time logical
opportunity into measured full-model acceleration, and how much do its sparse
paths contribute? Benchmark K050, K049, P0, K050 with sparse skipping disabled,
and K050 with attention skipping disabled across all35 retained checkpoints.
No checkpoint or intervention changes. All525 final processes use the same
physical RTX5090, BF16 B1/T2048 full50304-logit workload, native PyTorch/SDPA
CUDA-graph denominator and complete338-block qualification as Observation01.
Every point equally pools1344 paired ratios across three fresh processes.

The predictor is the retained canonical FP16 R_model, computed from pooled
integer logical-product counts over the full validation stream. It is not
recomputed from the BF16 timing run. Fit unweighted checkpoint-level OLS with
an estimated intercept to all35 qualified K050 points; do not force a1x intercept.
Timing uses64 fixed validation inputs; qualification and untimed diagnostics
cover all338 blocks/all500 documents, with the1444-token excluded tail retained.

## Result

K050 qualifies on all35 checkpoints (105/105 final processes). Speedups range
1.005938-1.783175x, with equal-checkpoint geometric mean1.250205x. The smallest
gain is near the noise floor and should not be advertised as a robust improvement.

    S_hat = 0.9531843003 + 3.8586397669 R_model
    R_squared = 0.7812571630; n = 35; R_model is a fraction.

The positive association supports workload-specific realization of sparsity,
not an exact or universal speed law. For example, c30 has canonical
R_model0.274826843 and measured1.783175x, below the fitted2.013642x.
The fitted line and all35 points remain visible; no checkpoint is discarded
to improve R_squared.

| Final implementation | Qualified checkpoints | Geomean vs native | c30 speedup |
|---|---:|---:|---:|
| K050 | 35/35 | 1.250205x | 1.783175x |
| K050, attention skipping disabled | 35/35 | 1.267131x | 1.802220x |
| K050, all sparse skip paths disabled | 35/35 | 1.183457x | 1.360005x |
| K049 | 35/35 | 1.123355x | 1.528395x |
| Sakana-derived Pythia adapter P0 | 5/35 | 0.867570x on those five only | 0.817961x |

At c30 the separate-process geometric means of median host latency are
0.844124ms native and0.473149ms K050; the reported speedup itself is computed
from raw paired ratios, not the ratio of these latency summaries.

## Sparse-path attribution and executed work

K050 / no-skip normalized-speedup ratios have geometric mean1.056401 across
all35 checkpoints. The sparse policy helps19 and hurts16, ranging0.953537 to
1.354226. At c30 the ratio is1.311153:31.1% additional acceleration relative to
the already fused no-skip implementation. Therefore fusion and sparse-path
specialization both contribute; the entire1.250x average gain cannot be
attributed to skipped multiplication.

Disabling attention skipping is faster on every checkpoint. K050 divided by
the attention-dense control averages0.986642 (range0.982900-0.989432). Here
"attention-dense" retains the custom attention kernel but disables its skip
branch; it is not a substitution with stock SDPA. Projection-side sparse/hybrid
specialization supplies the favorable net sparse-path contribution. This
observation does not turn the ablation into a newly searched winning candidate
or replace K050 in the historical progress curve.

All35 final diagnostic passes completed. At c30, the BF16 scalar opportunity
lower bound is0.274745375, distinct from canonical FP16 R_model0.274826843.
Pooling execution counters across layers before division gives:

| Diagnostic at c30 | Fraction |
|---|---:|
| QK MMA atoms skipped | 57.9952% |
| PV MMA atoms skipped | 67.2084% |
| h projection MMA atoms bypassed | 94.4975% |
| z projection MMA atoms bypassed | 99.6449% |

These counts establish that sparse branches execute, but skipped atoms do not
guarantee lower latency: attention skipping still loses. Projection MMA bypass
includes SIMT substitution (189838464 h and51767296 z scalar products), and
the h/z path has twofold row padding. These are not pure removed-FLOP fractions.

The controls are separate randomized fresh-process pairs against native.
Ratios between their normalized speedups are not direct within-process toggles.
Turning skipping off changes hybrid SIMT selection as well as zero detection;
this is an implementation-path ablation, not a pure zero-product intervention.

## Sakana comparator qualification

P0 qualifies at c01,c08,c20,c30,c35. The remaining30 checkpoints fail the
predeclared elementwise logit bound in all three processes. All those outputs
are finite, satisfy relative-L2 (largest0.003926) and pooled-loss bounds
(delta between-0.00007634 and0.00013667). Thus "unqualified" must not be
misreported as execution failure or a large loss degradation. Four of the five
qualified cases are slower than native; c01 is its native fallback at1.001212x.
No35-checkpoint qualified P0 mean exists. This is the existing adaptation, not
unchanged Sakana SparseLM/H100; see [Observation04](04-sakana-compatibility.md).

## Figure, caption and limits

Current paper asset: [03-matched-rmodel-speedup-r02.pdf](../figures/03-matched-rmodel-speedup-r02.pdf).

**Caption:** Full-model acceleration from the frozen final K050 policy versus
canonical train-time R_model across35 Pythia14M checkpoints on one RTX5090.
Uniform points are geometric means of matched native/candidate timing ratios
from three fresh processes, all passing complete validation. The line is
unweighted OLS with estimated intercept; its equation and R_squared are shown.
Axes are linear, with no empty0-1 speedup region and no variant legend.

This is one architecture, GPU, precision, workload and training seed. Checkpoints
differ in trained weights, topology and task loss; it is not a causal isolation
of R_model or a quality-matched model ranking. The denominator is the declared
native graph, not eager PyTorch or a search over strongest compiled-dense
baselines. Correlation and path ablations support a conditional systems claim,
not that R_model alone explains all runtime or that every sparse site helps.
No manuscript text or promoted finding is changed.

Source scripts: `02_benchmark.py`, `10_reduce.py`, `12_figures.py --revision 2`,
`16_report.py`; frozen Run028 `115_hybrid_diagnostics.py` supplies counters.
Data: `results/matched-retrospective-001.json`, `results/report-001.json` and
their hashed raw-source records. All35 full diagnostic JSONs remain local.
Figure hashes: `results/figures-002.json`. Revision1 is a superseded proof;
revision2 keeps the complete fitted line in view and corrects tick formatting,
without changing data, regression or exclusions.

## Approved manuscript clarification: attention and sequence length

On 8 September the user requested a brief discussion of whether longer T
could make attention skipping profitable. The manuscript subsection now
explicitly reports the executed MMA skips above and the negative attention
skip contribution at T=2048. The positive net sparse-path contribution is
projection-side; zero-product avoidance is not itself a latency result.

The longer-T discussion is an untested hypothesis based on the declared
full-sequence count equation: QK+PV work grows as `d*T*(T+1)` per block,
whereas projection and LM-head work are linear in T at fixed architecture.
This changes attention's share of logical work, not automatically its
runtime speedup. Frozen K035 `sparse_gemm.h` repeats zero checks/warp votes
within attention tiles; their cost can grow with attention work, and
fragment sparsity may change. Its `candidate.py` explicitly requires
`(1,4,2048,32)`, so longer sequences require kernel adaptation, a declared
model/context/data protocol, new numerical checks, logical measurements
and matched skip-on/off timings. No crossover or context-extension quality
result is claimed and no experiment was launched.

The draft remains local-only. Updated source snapshots are preserved in
`provenance/manuscript-20260908-r02/`, with verification recorded by
`21_verify_attention_discussion.py`. The existing figure and numerical
results are unchanged.
