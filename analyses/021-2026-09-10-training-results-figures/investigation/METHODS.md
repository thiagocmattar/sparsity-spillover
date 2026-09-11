# Evidence, kernel trace and field definitions

## Cohort and provenance

[Analysis 018 figure data][a18] selects the 30 K050 points c01–c30 and their
canonical integer accounting. The [Run 029 retrospective][retrospective] supplies
qualified replicate identities, diagnostic provenance and raw-file hashes.
The manuscript supplementary data's `runtime` object is exactly equal to
Analysis 018's runtime object; its full file is a separate snapshot, so the
whole-file hashes need not match. Its own `SOURCES.json` hash is checked.

The extraction verifies 270 raw timing files (30 checkpoints × three
implementations × three processes) and 30 diagnostics against retained hashes.
Archived implementation files are also checked against the archive manifest.
`data/provenance.json` records 321 source hashes, including the present replay
script. These are existing artifacts; this investigation executes no model.

Each diagnostic pools 338 complete 2,048-token blocks from the 500-document
validation set, covering 692,224 input tokens and excluding the 1,444-token tail.
Each checkpoint has six layer records. Integer counts are summed before division.
The diagnostic is an untimed counting pass attached to the first K050 replicate,
not a sample of counters from the timed 64-block subset.

## Dispatch and native-reference cost

[Run 029 `final_row`][replay] selects K050 with prefix shortcut disabled. The
all-skips-off control sets both attention `skip=False` and
`projection_skip=False`; attention-dense sets only attention `skip=False`.
All retain the same fusion. K050 inherits K049's joint FFN-down/output kernel,
K042's QKV/FFN-up kernels and K035's segmented attention implementation. K050
adds a shared-input, distinct-affine LayerNorm pair with existing a/m gates.
See the [K050 installer][k050], [K049 installer][k049py],
[K042 installer][k042py] and [K035 installer][k035py].

The native [attention forward][pythia] applies q_post, k_post and v gates
separately. [FixedSymmetricThreshold.forward][sites] returns its input for κ = 0;
otherwise it performs absolute value, comparison and `masked_fill`.
The [compatibility adapter][adapter] maps absent Q/K/V gates to zero thresholds;
the inherited [RopeGate wrapper][ropepy] and [CUDA kernel][ropecu] combine gates,
partial RoPE and head-major layout. This gives a concrete execution difference
consistent with the nonzero-κ native timing gap. A controlled per-operation
attribution is not present for this entire cohort.

The customary label "fusion only" means the **all-skips-off K050 control**. It
also includes its dense implementations and layouts, so it does not isolate the
cost of fusion as a single implementation change.

## Projection units and counters

### QKV and FFN-up: a/m input fragments

The [K042 projection kernel][projection] uses 32×64×64 thread-block tiles,
16×32×64 warp tiles and 16×8×16 MMA atoms. The MMA operator applies a warp-wide
exact-zero test to the entire activation A fragment (ignoring the zero sign
bit). It skips the MMA if all 16 tokens × 16 input features are zero. It does
not require a zero row, nor test the weight B fragment.

The original [diagnostic `projection_counts`][diagnostic] reconstructs this
decision on actual BF16 tensors, grouping them into 16×16 A fragments. Each zero
fragment bypasses `N/8` output MMA atoms: 48 for QKV (N = 384), 64 for FFN-up
(N = 512). These two count families are exact operand-derived reconstructions
of the stated predicate, **not direct CUDA instruction counters**. The fraction
of zero A fragments equals the corresponding MMA bypass fraction because N is
constant within each projection family.

### FFN-down and attention output: hybrid row/fragment execution

The [K049 joint kernel][joint] processes eight token rows and 128 output
features, with K = 512 for h/FFN-down and K = 128 for z/attention output.
Safe rows with at most two nonzeros use `short_linear`: scalar/SIMT products
over their live inputs, with bias/residual handling retained. Safety requires
finite weights and nonzero magnitudes between 2^-50 and 2^50, with the analogous
activation check. Remaining rows contribute a union of live K/16 tiles across
the eight rows. Empty tiles bypass 16 MMA atoms across the 128 outputs.

The executed MMA shape is 16×8×16, but only eight token rows are real; the upper
eight rows are padded zeros. Potential counts are 131,072 FFN-down and 32,768
attention-output MMA atoms per layer/block. They satisfy
`potential × 1024 = unpadded dense scalar products`.

Instrumented fields are h/z issued MMAs, bypassed MMAs and SIMT products. The
original diagnostic independently reconstructs these values from operands and
checks equality for every layer/block. Bypassed MMAs include both empty support
and support handled by SIMT; they cannot be labeled eliminated arithmetic.
For every checkpoint, row histograms imply exactly the recorded SIMT product
count: `128 × (one-nonzero rows + 2 × two-nonzero rows)`. The zero/≤2 row fractions
are histogram-derived opportunities, not separate recorded row-hit counters.

### Projection aggregation

`projection_mma_bypass_fraction` pools bypassed/potential MMA atoms across all
four families. This is a count of equally shaped hardware instructions, including
the padded h/z atoms and their SIMT substitutions. Family-specific fractions
remain available; pooling does not assert equal time per instruction.

For a secondary unpadded arithmetic-support proxy, the numerator is:

`2048 × (QKV + FFN-up bypasses) + 1024 × (FFN-down + output bypasses) − SIMT products`.

Divide by dense projection scalar products. This adjusts for h/z row padding
and remaining scalar work, but excludes memory, detection, launch and other
costs. It is neither measured hardware FLOPs avoided nor runtime savings.

## Attention units and counters

The [K035 sparse MMA helper][attention] tests complete warp-distributed A and B
fragments. An individual 16×8×16 MMA is skipped if **either** is entirely zero.
For QK, these are a 16×16 Q fragment and 16×8 transposed-K fragment; for PV,
a 16×16 probability fragment and 16×8 V fragment. Scattered zeros inside
otherwise live fragments do not suffice, and both operands need not be zero.

Each QK/PV family has 147,456 eligible MMA atoms per layer/block in this
implementation, or 299,040,768 over 338 blocks and six layers. "Eligible" here
means the instruction opportunities in the executed tiled loop, **including
causal padding**, rather than only unmasked logical products. Prefix shortcut
counts are checked to be zero. The instrumented QK and PV issued/skipped totals
are retained separately and as a combined instruction-weighted fraction.

The common 10.4167% PV skip fraction reflects masked/padded probability fragments
even when V is dense. Consequently it must not be interpreted as learned
unmasked scalar sparsity. Attention has only two distinct combined skip fractions
in this cohort: c30 and all other checkpoints.

## Scalar accounting and timing estimands

For each of the six operations, the canonical FP16 record supplies integer
`product_count` and `zero_product_count`. The operation scalar fraction uses its
own product denominator. The operation contribution in percentage points uses
the common model denominator, including the dense output head. The six
contributions sum to S_model. Projection-only groups QKV, attention output,
FFN-up and FFN-down; attention-only groups QK and PV.

Actual-BF16 diagnostics separately retain activation-zero lower bounds per
operation. In particular, the PV lower bound counts V zeros and omits
probability underflow; it is not the full canonical PV quantity. Primary
association plots use canonical FP16 scalar fractions and BF16 kernel counters
without conflating these measurements.

Timings preserve the retained graph-captured, uncached B1/T2048 BF16 protocol,
including full-vocabulary logits and the shared native graph-forward scaffold.
Each process records 64 selected validation inputs × seven paired passes, with
one native and one candidate host latency per pair. Output shape, complete pair
coverage, block identities, qualification and positive finite latencies are
checked. Geometric means across all 1,344 observations per mode/implementation
give `N`, `D`, `P`, `F`: native from the full-K050 processes, all-skips-off,
projection-on/attention-dense and full K050, respectively.

The primary requested quantities are `N/D`, `D/P`, `P/F`, `N/F`. Their product
telescopes exactly, and `N/F` reproduces the retained paired-GM speedup. These
are **geometric-mean host latencies**, not ratios of the older summary medians.
The three implementations were timed in separate fresh processes with their
own interleaved native control. All three native GMs and paired speedups are
retained. `*_native_normalized` fields additionally divide those separately
native-normalized speedups, matching the earlier figure's estimand; their
denominators differ slightly. Native drift relative to the full-K050 reference
is at most 0.1232% across these comparisons.

## CSV field guide

`checkpoints.csv` has 30 rows and 140 fields. Percent values are 0–100; fractions
are 0–1; contributions use percentage points (pp); latency is milliseconds.

| Field group | Meaning |
|---|---|
| condition, recipe, family, pressure, kappa, pressure_weight, checkpoint_evidence_id, validation_loss | Exact cohort identity, threshold/pressure metadata and canonical endpoint loss; κ is blank for local recipes. |
| model_products, s_model_percent | Common dense denominator and canonical model-wide zero-product percentage. |
| `{op}_products`, `{op}_zero_products`, `{op}_scalar_zero_fraction`, `{op}_model_contribution_pp` | Six operation numerators and their distinct operation/model denominators. |
| `{op}_bf16_*_lower_bound` | Separate BF16 diagnostic scalar counters/fractions, not canonical replacements. |
| `projection_*`, `attention_*` scalar/product fields | Integer-pooled projection-only and QK/PV-only accounting. |
| `{projection}_mma_issued/bypassed/potential`, `*_mma_bypass_fraction` | Four projection families, plus combined totals/fraction. |
| `qkv/ffn_up_zero/total_16x16_fragments` | A-fragment counts reconstructed from original diagnostic MMA counts. |
| `{projection}_rows/zero_rows/rows_le2`, `*_zero_row_fraction`, `*_row_le2_fraction` | Actual BF16 input-row histograms, pooled across layers/blocks. |
| `*_simt_products`, `*_simt_opportunity_products`, `*_simt_safety_gap_products` | Executed scalar products and histogram reconstruction; safety gap is zero throughout this cohort. |
| `qk/pv_mma_issued/skipped/potential`, `*_mma_skip_fraction`, combined `attention_mma_*` | Instrumented attention instruction counts, including masked/padded work. |
| `all_skips_off/projection_on/full_native_gm_ms/candidate_gm_ms/paired_speedup` | Reconstructed raw timing GMs and original-estimand paired speedups for each implementation. |
| `native_baseline_gm_ms`, `fusion_only_speedup_common_native`, `projection_sparse_gain`, `attention_sparse_gain`, `full_speedup` | N, N/D, D/P, P/F and N/F, as defined above. |
| `*_native_normalized`, `projection_arithmetic_avoidance_proxy_fraction` | Explicit secondary estimands, not substitutes for the primary gains. |
| `projection_mma_counter_method`, `diagnostic_source` | Counter provenance and retained per-checkpoint source. |
| Four blank unavailable fields below | Missing measurements are empty in CSV and null in JSON, never zero. |

`layer-counters.csv` retains the 180 layer-level instrumented records.
`matched-pairs.csv` retains both values, A7−A4 differences and relevant ratios
for all ten pairs. `associations.csv` reports n, unique predictor values,
Pearson r, OLS slope/intercept/R² and undefined-status explanations.

## Missing quantities and minimal measurement changes

| Unavailable quantity | Minimal change needed; not executed here |
|---|---|
| Pure-zero-only projection bypass versus SIMT substitution | In K049's count-enabled joint path, record each original tile-support mask before simple-row removal and the remaining mask afterward. Count original-empty and emptied-by-SIMT tiles separately; optionally count actual fast rows. Marginal histograms cannot recover these joint tile masks. |
| Projection/attention counters on the timed 64 inputs | Add explicit validation block indices to `115_hybrid_diagnostics.collect`, replacing its contiguous `range(blocks)` selection, and persist counters per input. Run an untimed pass on the retained timing indices. Existing aggregates cannot be subsetted. |
| Unmasked-only eligible attention MMA fraction | Add count-only classification by Q/K token coordinates and causal mask in K035. Record padded-only, mixed-validity and fully valid instruction opportunities and their skip counts separately. Mixed atoms cannot be fractionally treated as instructions actually omitted. |
| Isolated native Q/K/V gate cost or per-family runtime benefit | Add per-operation CUDA/NVTX profiling for the same checkpoints and timing inputs; for attribution, compare a same-checkpoint native fused-gate control and family-specific projection switches under the same timing protocol. Full-model two-switch ablations do not identify each operator's latency. |

The four null fields are `projection_pure_zero_only_bypass_fraction`,
`projection_timing_subset_bypass_fraction`, `attention_unmasked_only_mma_skip_fraction`
and `attention_timing_subset_mma_skip_fraction`. The existing data suffice for
the reported cohort comparisons; these additional measurements would narrow
specific attribution/coverage limits, without requiring new training.

[a18]: ../../../analyses/018-2026-09-08-results-materials/figure_data.json
[retrospective]: ../../../runs/029-2026-09-07-pythia14m-matched-kernel-retrospective/results/matched-retrospective-001.json
[replay]: ../../../runs/029-2026-09-07-pythia14m-matched-kernel-retrospective/replay.py#L48
[diagnostic]: ../../../runs/029-2026-09-07-pythia14m-matched-kernel-retrospective/archive/root/runs/028-2026-09-06-pythia14m-all-site-sparse-kernels/115_hybrid_diagnostics.py
[k050]: ../../../runs/029-2026-09-07-pythia14m-matched-kernel-retrospective/archive/root/runs/028-2026-09-06-pythia14m-all-site-sparse-kernels/candidates/k050/candidate.py
[k049py]: ../../../runs/029-2026-09-07-pythia14m-matched-kernel-retrospective/archive/root/runs/028-2026-09-06-pythia14m-all-site-sparse-kernels/candidates/k049/candidate.py
[k042py]: ../../../runs/029-2026-09-07-pythia14m-matched-kernel-retrospective/archive/root/runs/028-2026-09-06-pythia14m-all-site-sparse-kernels/candidates/k042/candidate.py
[k035py]: ../../../runs/029-2026-09-07-pythia14m-matched-kernel-retrospective/archive/root/runs/028-2026-09-06-pythia14m-all-site-sparse-kernels/candidates/k035/candidate.py
[projection]: ../../../runs/029-2026-09-07-pythia14m-matched-kernel-retrospective/archive/root/runs/028-2026-09-06-pythia14m-all-site-sparse-kernels/candidates/k042/projection.cu
[joint]: ../../../runs/029-2026-09-07-pythia14m-matched-kernel-retrospective/archive/root/runs/028-2026-09-06-pythia14m-all-site-sparse-kernels/candidates/k049/joint.cu
[attention]: ../../../runs/029-2026-09-07-pythia14m-matched-kernel-retrospective/archive/root/runs/028-2026-09-06-pythia14m-all-site-sparse-kernels/candidates/k035/sparse_gemm.h
[pythia]: ../../../runs/029-2026-09-07-pythia14m-matched-kernel-retrospective/archive/root/src/sparsity_research/pythia.py#L215
[sites]: ../../../runs/029-2026-09-07-pythia14m-matched-kernel-retrospective/archive/root/src/sparsity_research/sites.py#L100
[adapter]: ../../../runs/029-2026-09-07-pythia14m-matched-kernel-retrospective/archive/root/runs/027-2026-09-06-pythia14m-kernel-sparsity-characterization/adapter.py#L24
[ropepy]: ../../../runs/029-2026-09-07-pythia14m-matched-kernel-retrospective/archive/root/runs/026-2026-09-06-pythia14m-fused-sparse-kernel/autoresearch/candidates/k019/candidate.py#L24
[ropecu]: ../../../runs/029-2026-09-07-pythia14m-matched-kernel-retrospective/archive/root/runs/026-2026-09-06-pythia14m-fused-sparse-kernel/autoresearch/candidates/k019/kernel.cu#L10
