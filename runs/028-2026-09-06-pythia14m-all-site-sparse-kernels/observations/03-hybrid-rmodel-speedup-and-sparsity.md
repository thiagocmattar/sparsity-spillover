# R_model and measured acceleration with the uniform K050 kernel

## Question

Can a more specialized kernel turn the retained activation sparsity into
larger, measurable full-model savings, and does the benefit increase at the
higher-R_model variants? The question is about these existing checkpoints,
not new training, additional pruning, or a universal speedup law.

## Sources, method, and coverage

Frozen policy: `../final-policy-002.json`, SHA256
`f4b390ecf76e63ba8b155738ee0748e8f46820e4e2df7e89dda4e07afc00bb40`.
All 61 frozen sources and 900 dependencies retain their registered identities.
The uniform K050 implementation was frozen after all ten registered
development endpoints and the primitive checks; two subsequent harness
smokes preceded the final study. There is no per-checkpoint kernel selection.

All 35 checkpoints are existing random-initialization pretraining endpoints,
seed 1234, step 712, identified by Run027 `prelaunch/inputs.json`. The source
gates, thresholds, weights and data remain unchanged. Historical A4+OL1@h is
retained and labeled separately from four-site pressure. The 248 original
input/provenance files, including all checkpoints, were reverified locally
before Pod deletion.

Raw evidence: `../artifacts/final-matrix-002/` and all
`../artifacts/final-cNN-pR-002/` folders. All 105 processes completed in
4,803.876 seconds with no infrastructure failures. The evidence-050 archive
has SHA256
`c6ac55769cf9ba747f1e57d352eef6f071cc60f4249134545a4938ae4d9f4726`.
Its complete inventory of 19,738 files / 709,108,815 bytes, including earlier
terminal trials, was verified locally before teardown.

Hardware: one RTX 5090 32 GB; PyTorch 2.11.0+cu128, CUDA 12.8, BF16, batch 1,
uncached causal sequence length 2,048, full 50,304-vocabulary logits. Input
staging, compilation and graph capture are outside warm timing. There are
three fresh processes per checkpoint, 64 fixed validation input identities
(seed 2504), and seven paired passes per process. Checkpoint order is
randomized within each replicate using seed 2504. Nine modes are retained:
native eager/graph, previous eager/graph, new eager/graph, sparse paths disabled,
attention skips disabled, and unfused K049 graph. No unfavorable measurements
or numerical failures are excluded.

Each process validates all 338 complete blocks from all 500 MiniPile validation
documents: 692,224 input tokens, 691,886 prediction tokens, 1,444-token excluded
tail. Instrumented diagnostics cover all 338 blocks once per checkpoint.
Every new mode and its four controls passes the fixed numerical gate:
atol 0.25, rtol 0.02, relative L2 <= 0.02, absolute pooled loss change <= 0.001
nat/token. This is tolerance qualification, not universal exact equivalence.

K050 has zero observed logit/loss discrepancy at 33 of 35 checkpoints. At c33,
the maximum logit difference is 0.25, relative L2 0.0002096063, and absolute
pooled loss change 5.278823e-7 nat/token; at c34 these are 0.125, 0.0001596486,
and 1.411450e-9. In replicate 1 the affected blocks are c33:107 and c34:118/259.
These discrepancies also occur in unfused K049 and attention-dense controls,
but not the no-skip control; they cannot be attributed to the new paired
normalization fusion alone. The results refute universal numerical identity
while satisfying the predeclared accuracy limits. No tolerance was relaxed.

Canonical R_model is pooled source FP16 logical zero-product opportunity,
with the dense LM-head denominator. Runtime and diagnostic operands use BF16.
These are distinct quantities. `../121_reduce_hybrid.py` audits provenance,
frozen source snapshots, complete validation, numerical gates, all nine modes'
paired sample coverage and integer counter conservation. Its output is
`../results/summary-002.json`. `../122_hybrid_figures.py` creates the PDFs,
`../results/per-variant-002.md`, and `../results/figure-provenance-002.json`.
The complete table includes native validation NLL to expose quality differences.

## Figure caption and legend

**Figure 03. Full-model acceleration and matched sparsity-aware contribution
across all 35 Pythia-14M variants.** (a) Native graph latency divided by K050
graph latency. (b) The same K050 implementation with sparsity-aware paths
disabled divided by latency with those paths enabled. Both sides of (b)
retain the paired-normalization fusion. A ratio above one favors enabled
paths. Panel (a) starts at zero; panel (b) has an explicitly expanded scale.
Colors and shapes identify activation/pressure families. Points are geometric
means of paired ratios, equally weighted across three processes; bars span
the three process geometric means, not confidence intervals or training-seed
uncertainty. Every comparison has 1,344 paired observations per variant.
All 35 new variants qualify on full validation in every process. Canonical
R_model uses FP16 logical products, while execution uses BF16. The h/z disabled
paths use a twofold row-padded MMA fallback; the ablation therefore measures
the complete hybrid sparse-path policy, not only zero-instruction removal.

Output: `../figures/03-hybrid-rmodel-speedup-and-sparsity.pdf`.

## Results

| Variant | R_model (%) | Native/new graph | Paths disabled/enabled | Previous/new graph |
|---|---:|---:|---:|---:|
| A0, c01 | approximately 0 | 0.978736 | 0.943986 | not qualified |
| A4, kappa 0.5, c15 | 10.215537 | 1.508222 | 1.285691 | 1.088381 |
| A4+OL1@4, kappa 0.5, c20 | 12.713449 | 1.571169 | 1.338247 | 1.123186 |
| A7, kappa 0.5, c25 | 15.386813 | 1.720621 | 1.289681 | 1.088129 |
| A7+OL1@7, kappa 0.5, c30 | 27.482684 | 1.738990 | 1.305115 | 1.101296 |

The full-model relationship is broadly positive, but not universally
monotonic. At the highest R_model endpoint, c30, native/new graph ranges
1.738840-1.739142 across processes, and sparse-path disabled/enabled ranges
1.304845-1.305585. Its pooled median graph latencies are native 0.822932 ms,
K050 0.473044 ms, previous 0.520874 ms, and paths-disabled 0.617873 ms.
Ratios in the figure are paired geometric means, not ratios of these medians.

Eighteen of 35 variants have a positive net sparse-path ratio. Low-sparsity
A0 regresses against native graph and against the paths-disabled control;
this point remains visible. Higher scalar opportunity is not a sufficient
ordering statistic: c20 has a larger sparse-path ratio than c30 despite
lower R_model. Within A7+OL1@7, c26-c30, increasing R_model accompanies graph
ratios 1.141917, 1.298743, 1.303999, 1.389166, 1.738990; sparse-path ratios are
0.972401, 0.972867, 0.976729, 1.041095, 1.305115. This supports a useful
high-sparsity benefit without turning every small R_model increase into a gain.

At c30 the mean paired native-to-new saving is 0.349865 ms; enabling the
sparse paths saves 0.144453 ms relative to the matched disabled-path policy.
The ratio of these mean differences is 41.29%. This is a descriptive
within-implementation ablation, not a universal fraction of dense runtime
explained by R_model: it includes hybrid dispatch, padded fallback and
interactions. A 1.305115 ratio corresponds to about 23.38% less latency than
that disabled-path control, not a 30.51% latency reduction. Fusion is measured
separately in observation 04; gains must not be added as independent effects.

## Caveats and supported claim

The supported claim is that **a specialized implementation can obtain larger
full-model speedups at these high-R_model endpoints, with a substantial
positive matched sparse-path benefit**. It does not capture all logical
zero products, does not make attention skipping profitable, and is not an
equal-quality frontier. For example, c30 native NLL is 5.8313 nat/token versus
A0's 5.2100. Different checkpoints also differ in topology, weights and gate
costs. The within-checkpoint ablation is stronger attribution evidence than
the cross-variant scatter alone. One training seed, one GPU, one batch size
and one sequence length are covered; there is no causal regression on R_model.

The new study is separate from the earlier frozen K036 study and preserves
its results. No cross-study unpaired latency ratio is presented as a matched
benchmark. No finding promotion or manuscript TeX modification is made.
Both PDFs are vector output with embedded TrueType fonts; their complete
150-dpi renders were visually inspected without clipping or overlap.
