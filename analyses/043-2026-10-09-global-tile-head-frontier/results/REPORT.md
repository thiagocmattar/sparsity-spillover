# 14M tile/head quality–latency and pressure geometry

The 14M-only inventory contains **87 execution points** and **12 observed
nondominated points**. None of the 18 head execution points extends this observed
14M envelope. The specialized 14M T2 corner-tile model at κ=0.1 does appear on
it: loss **5.596746**, complete-forward **0.486441 ms** (Run064). This is a descriptive
comparison of retained measurements across sessions and implementations.

The head-aligned κ=0.5 kernel is **3.10% faster** than its own native dense execution,
but its loss is **5.882600**. Within the FP16 native-body/head cohort, the Head16 κ=0.1
native dense execution has both lower loss (5.560725) and lower point-estimate
latency (0.983987 ms) than Head16 κ=0.5 sparse (0.990146 ms). More bypass therefore
does not establish a better measured quality–latency trade-off in this inventory.

## Scope and comparability

At the user's request, the cohort selects only the 14M coordinates from the
previous inventory. It includes Analysis037's 14M historical study, Analysis041's
T2/T7 tile experiments and matched scalar references, and the executions in
Runs063/064/067/068. No quality or latency coordinate changes. It is a cohort of existing
evidence, rather than an exhaustive inventory of every repository experiment.
Different executions of the same retained checkpoint remain separate points.

All quality coordinates use canonical final-checkpoint full validation: 500
MiniPile documents, 338 complete 2,048-token blocks, 692,224 input tokens and
691,886 shifted predictions; the 1,444-token tail is excluded. Timing is RTX 5090,
B1/T2048 with all 50,304 vocabulary logits for every token. It is full-sequence
forward, not cached decoding or token sampling. Loss evaluation/softmax is outside
the vocabulary-projection timing. Native B1 qualification losses and BF16 kernel
anchor losses are not substituted for canonical FP16 quality coordinates.

Older optimized block executions use BF16. Runs067/068 use FP16, native dense
transformer projections, and a separately optimized head. No measurement combines
that head kernel with the earlier sparse block pipeline. There is no latency
rescaling, normalization to Base, extrapolated combined speedup, or gate-off
ablation. A checkpoint's trained gates remain active in native and no-skip modes.

Figures [01](../figures/01-global-quality-latency.pdf) and
[02](../figures/02-execution-cohort-frontiers.pdf) show both the global envelope and
the separate execution cohorts. There are 69 BF16 block executions (12 frontier
points) and 18 FP16 head executions (3 frontier points, all native dense). Session
differences remain within those cohorts. Frontier membership uses strict Pareto
dominance of point estimates, not significance tests; process ranges are not CIs.
The dotted step envelope does not assert measured intermediate executions.
Figure01 uses one 14M-only panel with linear axes and connected historical T2/Ph
and T7/Ph sweeps, following Analysis041 Figure04 panel (b). The latency range
expands to include the new head modes; none are cropped from view.

## What is tiled and what runs sparsely?

| Evidence | Trained body gate | Trained head gate | Measured sparse execution |
| --- | --- | --- | --- |
| Historical scalar T2 / Ph | h,z, elementwise | OFF | 14M h/z consumer |
| Historical scalar T4 / T7 | Declared 4 / 7 sites, elementwise | OFF | Site-specific kernel package; a gate alone does not imply bypass |
| T2 corner models, Runs060/061/063/064 | h,z, 8×16 corner tiles | OFF | Generic or corner-aware h/z consumer |
| T7 corner models, Run062 | Declared seven sites, 8×16 corner tiles | OFF | Retained T7 consumer package |
| Run066 training / Run067 timing | h,z, 8×16 tiles, κ=0/0.1 | 8×16 tiles, same κ | Head only; consumer M16/K16 requires two neighboring token tiles empty |
| Run068 | h,z, 8×16 tiles, κ=0 | 16×16 tiles, κ=0.1/0.5 | Head only; gate and consumer tiles aligned |

Tiles mean **tokens × input features**, not vocabulary-output columns. The trained
head gate acts on `head_in.final`, after learned affine final LayerNorm and before
the vocabulary matmul. It retains an entire tile iff its signed upper-left corner
is greater than κ; rejected tiles are zero, retained signed interiors are unchanged.
There is no gate or pressure on vocabulary weights or output logits. Ph means OL1
at post-gate h; Ph+head also includes the final post-gate head-input activation.

## New head points: full latency and quality

All times are synchronized host geometric means in milliseconds, pooled over
three fresh processes/checkpoint and 64 timing inputs × 7 paired passes/process.
The sparse-versus-dense column is a latency change: a positive value is slower.
Skip savings use the same custom kernel with skipping disabled as denominator.
Canonical loss is measured in the original training evaluator (B32), whereas B1
kernel qualification compares all complete validation blocks. All six head
checkpoints' custom logits and B1 loss match native exactly in their source runs.

| Condition | Canonical loss | Dense ms | No-skip ms | Sparse ms | Sparse vs dense | Skip saves vs no-skip |
| --- | --- | --- | --- | --- | --- | --- |
| Head8 / Ph / h,z κ=0 / head κ=0 | 5.529001 | 0.998697 | 1.127280 | 1.074223 | +7.56% | 4.71% |
| Head8 / Ph+head / h,z κ=0 / head κ=0 | 5.497717 | 0.999435 | 1.127566 | 1.071257 | +7.19% | 4.99% |
| Head8 / Ph / h,z κ=0.1 / head κ=0.1 | 5.547977 | 0.999336 | 1.127475 | 1.087872 | +8.86% | 3.51% |
| Head8 / Ph+head / h,z κ=0.1 / head κ=0.1 | 5.507129 | 1.001191 | 1.129308 | 1.086614 | +8.53% | 3.78% |
| Head16 / Ph+head / h,z κ=0 / head κ=0.1 | 5.560725 | 0.983987 | 1.109741 | 0.994169 | +1.03% | 10.41% |
| Head16 / Ph+head / h,z κ=0 / head κ=0.5 | 5.882600 | 1.021812 | 1.148920 | 0.990146 | -3.10% | 13.82% |

Projection and head-path timings are separately measured boundaries, not additive
components of the independently timed full forward:

| Condition | Projection dense | Projection no-skip | Projection sparse | Head path dense | Head path no-skip | Head path sparse |
| --- | --- | --- | --- | --- | --- | --- |
| Head8 / Ph / h,z κ=0 / head κ=0 | 0.186104 | 0.312080 | 0.259049 | 0.202762 | 0.331052 | 0.279838 |
| Head8 / Ph+head / h,z κ=0 / head κ=0 | 0.187160 | 0.312445 | 0.254633 | 0.203757 | 0.331502 | 0.275828 |
| Head8 / Ph / h,z κ=0.1 / head κ=0.1 | 0.186076 | 0.312550 | 0.272871 | 0.202987 | 0.331294 | 0.293425 |
| Head8 / Ph+head / h,z κ=0.1 / head κ=0.1 | 0.186947 | 0.312645 | 0.269434 | 0.204100 | 0.332040 | 0.291100 |
| Head16 / Ph+head / h,z κ=0 / head κ=0.1 | 0.196628 | 0.318962 | 0.203997 | 0.214132 | 0.339314 | 0.223645 |
| Head16 / Ph+head / h,z κ=0 / head κ=0.5 | 0.197138 | 0.321092 | 0.162489 | 0.213253 | 0.339114 | 0.180095 |

The head path includes the gate, cast, and vocabulary projection. Full CUPTI stage
decomposition is **unavailable** for these six checkpoints. Figure
[04](../figures/04-head-measured-boundaries.pdf) presents the overlapping boundaries
separately. Figure [03](../figures/03-existing-full-decomposition.pdf) and the
[decomposition table](DECOMPOSITION.md) retain the complete, consistently colored
profiles from Run064. Their vocabulary heads have threshold OFF.

Pooled B1 occupancy/instruction counts over the same complete validation workload:

| Condition | Empty gate tiles | Empty M16/K16 consumer tiles | MMA bypass |
| --- | --- | --- | --- |
| Head8 / Ph / h,z κ=0 / head κ=0 | 25.324% | 14.380% | 14.380% |
| Head8 / Ph+head / h,z κ=0 / head κ=0 | 27.424% | 15.918% | 15.918% |
| Head8 / Ph / h,z κ=0.1 / head κ=0.1 | 17.336% | 7.075% | 7.075% |
| Head8 / Ph+head / h,z κ=0.1 / head κ=0.1 | 19.796% | 9.067% | 9.067% |
| Head16 / Ph+head / h,z κ=0 / head κ=0.1 | 39.881% | 39.881% | 39.881% |
| Head16 / Ph+head / h,z κ=0 / head κ=0.5 | 70.763% | 70.763% | 70.763% |

MMA bypass is omitted matrix instructions divided by potential instructions, not
a speedup. For the head8 models, two adjacent token tiles must both be empty before
an M16 consumer tile can skip. Alignment removes that mismatch in Run068. Native
training B32 occupancy can differ slightly from the benchmark's B1 count; these
rows deliberately use benchmark pooled integers throughout.

## Orthogonalization resistance: global Ph versus Ph+head

Figure [05](../figures/05-orthogonalization-head-comparison.pdf) shows all eight
retained tile/head training conditions. Figure
[06](../figures/06-manuscript-style-geometry.pdf) adds the original manuscript's
20 14M T4/T7 Ph/Pall conditions in its two-panel geometry view. Every condition
uses all 712 optimizer boundaries, including one initial zero-learning-rate
boundary. λ=b=1 and ε=10⁻¹². The eight recent conditions share original random
initialization and data schedule; one training seed per condition is available.

Let u be the task Adam direction and w the pressure gradient preconditioned with
the same task second moment, pooled over eligible parameters. Let
α=max(0,−u·w)/(‖u‖²+ε) and w_safe=w+αu in the conflicting case.

| Quantity | Exact meaning |
| --- | --- |
| Manuscript opposing component, 100ρopp | 100α, removed component norm relative to task-direction norm |
| Percent of pressure component removed | 100α‖u‖/‖w‖; norm fraction, not pressure energy or coordinates |
| Raw-gradient conflict | Frequency of g_task·g_pressure < 0 |
| Adaptive-direction conflict | Frequency of u·w < 0 with ‖u‖²>ε; equals projection_applied |
| Pressure norm before projection | ‖w‖, absolute adaptive L2 norm |
| Pressure/task norm before projection | ‖w‖/(‖u‖+ε) |
| Manuscript pre-cap ratio r/b | λ‖w_safe‖/(‖u‖+ε)/b, after projection and before norm capping |
| Cap active | Actual trust_scale < 1 |

100ρopp is not bounded by 100%; it has the task norm as denominator. The extra
pressure-normalized fraction is a different metric. The manuscript's r/b is also
different from the pre-projection norm ratio requested here; both are reported.
The table gives step medians for continuous quantities and pooled boundary counts
for conflict/cap frequency. Quartiles and full traces are retained in the data.

        | Condition | ρopp % | Removed pressure % | ‖w‖ before | ‖w‖/‖u‖ before | r/b after | Raw conflict | Adaptive conflict | Cap active | Eligible tensors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| No head \| hz=0 | 4.448 | 13.977 | 227.845 | 0.330 | 0.327 | 55.90% | 99.72% | 1/712 | 65 |
| No head \| hz=0.1 | 1.978 | 9.018 | 146.583 | 0.218 | 0.217 | 60.81% | 99.58% | 1/712 | 65 |
| H8 Ph \| hz=head=0 | 4.680 | 14.981 | 213.489 | 0.316 | 0.311 | 63.34% | 99.30% | 3/712 | 65 |
| H8 Ph \| hz=head=0.1 | 2.227 | 9.377 | 165.055 | 0.245 | 0.243 | 65.03% | 99.72% | 1/712 | 65 |
| H8 Ph+head \| hz=head=0 | 6.222 | 20.202 | 246.421 | 0.313 | 0.306 | 81.74% | 99.44% | 1/712 | 75 |
| H8 Ph+head \| hz=head=0.1 | 8.503 | 23.625 | 304.972 | 0.388 | 0.376 | 85.39% | 99.30% | 1/712 | 75 |
| H16 Ph+head \| hz=0, head=0.1 | 6.421 | 20.509 | 225.507 | 0.314 | 0.307 | 80.48% | 99.44% | 1/712 | 75 |
| H16 Ph+head \| hz=0, head=0.5 | 7.444 | 20.593 | 263.528 | 0.379 | 0.370 | 85.96% | 99.44% | 1/712 | 75 |

In the matched head8 κ=0.1 pair, adding head-input pressure changes median opposing
component from **2.227% to 8.503%** of task-direction norm, removed pressure component
from **9.377% to 23.625%**, and raw-gradient conflict from **65.03% to 85.39%**.
Adaptive projection is already frequent in both (**99.72% / 99.30%**), so conflict
frequency alone hides the larger opposing component. At κ=0, the matched changes
are **4.680%→6.222%** of task norm and **63.34%→81.74%** raw conflict.

Higher measured opposition does not imply worse quality in these pairs: Ph+head
has lower canonical loss than Ph by **0.031284 / 0.040847** at the same head8
κ=0 / 0.1. The global geometry is a diagnostic, rather than a standalone
predictor of the resulting quality or inference speed.

These recent conditions activate the norm cap in only **1–3 of 712 boundaries**,
including the initial zero-LR boundary. They do not reproduce the strong cap
saturation of the manuscript's T7/Pall context (3,453/3,560 boundaries, **96.99%**
pooled over its five thresholds). Pressure can face more conflict without reaching
the cap. Geometry at a zero-LR boundary does not imply a realized parameter move.

This is **global pressure geometry**, as confirmed by the user. There is no head-only
pressure run. Ph averages six h tensors with weight 1/6 each; Ph+head averages
those six plus one final head-input tensor with weight 1/7 each. The eligible pool
also expands from 65 to 75 parameter tensors, because head-input pressure reaches
more upstream parameters; vocabulary output weights are excluded. Consequently
these results cannot isolate the head's local gradient, assign its resistance to
a specific layer, or claim that only one site caused the difference. Comparison
of head8 with head16 additionally changes alignment and, at head κ=0.1, body κ.

## Observed 14M frontier

Sorted by increasing canonical loss; all native/skip-disabled executions were
eligible. Ties remain nondominated unless another point is strictly better in at
least one objective. A cheaper training intervention is not inferred from this
inference-only frontier.

        | Model | Training recipe | Body gate κ or pressure λ | Head κ | Execution | Loss | Full host ms | Session | 14M frontier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 14M | ReLU h / Ph | λ=0.5 | — | K050 | 5.110235 | 0.616329 | Run029 | yes |
| 14M | ReLU h / Ph | λ=1 | — | K050 | 5.121030 | 0.599598 | Run029 | yes |
| 14M | T2 scalar / Ph | 0.05 | — | K050 | 5.134379 | 0.590124 | Run041 | yes |
| 14M | T2 scalar / Ph | 0.1 | — | K050 | 5.151098 | 0.573427 | Run041 | yes |
| 14M | T2 scalar / Ph | 0.1 | — | sparse | 5.152346 | 0.565242 | Run061 | yes |
| 14M | T4 scalar / Ph | 0.05 | — | K050 | 5.195590 | 0.561801 | Run029 | yes |
| 14M | T4 scalar / Ph | 0.1 | — | K050 | 5.228687 | 0.529309 | Run029 | yes |
| 14M | T2 scalar / Ph | 0.5 | — | K050 | 5.536255 | 0.506323 | Run044 | yes |
| 14M | T2 tile / Ph | 0.1 | — | tile_sparse | 5.596746 | 0.486441 | Run064 | yes |
| 14M | T4 scalar / P0 | 0.5 | — | K050 | 5.659680 | 0.478819 | Run029 | yes |
| 14M | T4 scalar / Ph | 0.5 | — | K050 | 5.722666 | 0.462252 | Run029 | yes |
| 14M | T4 scalar / Pall | 0.5 | — | K050 | 6.037987 | 0.459476 | Run029 | yes |

[All 87 14M points](ALL-POINTS.md) · [All-execution CSV](all-executions.csv) ·
[Frontier CSV](global-frontier.csv) · [Head-boundary CSV](head-boundaries.csv) ·
[Geometry CSV](pressure-geometry.csv) · [Analysis scope and reproduction](../README.md).

This is a descriptive analysis of retained results. No new GPU run, manuscript
change, or research-finding promotion was performed.
