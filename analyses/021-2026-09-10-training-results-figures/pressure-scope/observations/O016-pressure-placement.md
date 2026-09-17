# O016 - Separating threshold and pressure placement

## Question and method

At fixed threshold scope and kappa, what changes when pressure goes from none
to h-only and then all sites? At fixed h-only pressure, what changes when
thresholding expands from four to seven sites? Which changes in local zeros
and executable structure accompany the endpoint trade-offs?

The user requested the targeted manuscript integration on 17 September.
[evidence.py](../evidence.py) joins the author-approved Analysis 023 table,
all corresponding activation/logical diagnostics and training logs, the
original Analysis 018 endpoints, and retained Run 029 timing records.
No training, checkpoint evaluation, benchmark or cloud launch was performed.

All 30 multisite conditions match initialization and data-order hashes, seed
1234, 712 updates, and 1,493,172,224 input tokens. Each final validation covers
500 documents, 338 complete 2,048-token blocks and the recorded 1,444-token
excluded tail. Integer counts are pooled before division. Site fractions and
logical-product counts are separately measured diagnostics; slight FP16
pass-to-pass differences are not forced to agree. Q/K are post-RoPE.
The historical Run 012 h-only identity follows the retained Analysis 009
execution audit, overriding its incorrect all-site declaration.

Reported losses preserve Analysis 023's approved values: h-only uses ordinary
final validation and the four older curves use eager logical-pass loss.
Both use the same final checkpoints and complete data. The largest discrepancy
is 0.000105806356 nats; each endpoint and contrast retains uniform-pass
alternatives in [the reduction](../data/evidence.json).

## Figures and captions

All figures are produced by [figures.py](../figures.py), with no changes to
the older analysis PDFs.

1. [Overview](../figures/01-overview.pdf): 30 multisite endpoints plus A0 and
   ReLU. Blue/orange means four/seven threshold sites. Open circles/solid,
   open diamonds/dashed and filled triangles/dash-dot mean no, h-only and
   all-site pressure. Dotted control paths are post-hoc thresholding. Eight
   local pressure endpoints remain in the complete table, not this overview.
2. [Pressure increments](../figures/03-pressure-increments.pdf): columns hold
   threshold scope fixed; top/bottom show loss/sparsity increments at matched
   kappa. Diamonds subtract P0 from Ph; triangles subtract Ph from Pall.
   Equal x spacing is categorical. Lines join trained conditions, not trajectories.
3. [OL1 geometry](../figures/04-ol1-geometry.pdf): medians over 712 steps of
   the removed opposing component as a percentage of the task-direction norm,
   and of pre-cap r/b. Four recipes retain both threshold scopes under h-only
   pressure. The cap panel pools 3,560 steps per recipe using logged s < 1.
   No intervals imply replicated-seed uncertainty. Coordinates precede group
   learning rates and weight decay; they do not guarantee task-loss preservation.
4. [High-threshold structure](../figures/05-site-structure.pdf): all six
   recipes at kappa=0.5. Seven-site exact-zero counters separate m and h;
   operation bars use the common model denominator, including the dense output
   projection. Rounded 0/100 labels are not assertions of exact boundaries.
5. [Cross-size](../figures/06-scale.pdf): retains the measured broad-pressure
   curves, absolute losses and architecture ceilings. All-site pressure has
   explicit labels and a consistent marker style; h-only transfer is absent.
6. [Quality and latency](../figures/07-quality-latency.pdf): all 35 qualified
   historical K050 endpoints, with absolute full-model latency on the left and
   matched projection gain versus MMA bypass on the right. Blue squares restore
   the five four-site h-only conditions. The native A0 cross is a reference,
   not another optimized checkpoint. Native-relative ratios are not the latency
   ranking. The descriptive 35-point fit is recomputed, not copied from n=30.
7. [Moderate-threshold structure](../figures/appendix-moderate-structure.pdf):
   the same six-recipe/site/operation comparison at kappa=0.05.
8. [Layer zeros](../figures/appendix-layer-zeros.pdf): two pages, kappa=0.05
   then 0.5. Cells pool validation counts at each of seven sites and six layers.
   Other thresholds are retained in the machine-readable reduction.

## Results

- At kappa=0.05, seven-site h-only gives S_model=10.1261% and loss 5.194959
  versus dense loss 5.2086: comparable at the evaluated seed.
- At kappa=0.5, expanding threshold scope at fixed Ph adds 6.4363 pp for
  +0.009383 loss. Broadening seven-site pressure from Ph to P7 then adds
  10.8190 pp for +0.097357 loss. These are different matched interventions.
- H-only cap activity is 9/3,560=0.2528% for each threshold scope. Broad
  four-/seven-site activity is 0.5056%/96.9944%. Median r/b is 0.1020/0.0860
  for h-only and 0.2533/57.4851 for broad four-/seven-site pressure.
- At kappa=0.5, expanding four-site pressure changes m zeros from 66.44%
  to 96.77%, with h about 99.94%. Expanding seven-site pressure changes
  pooled Q/K/V zeros from 29.52% to 95.60%, with m changing 67.41% to 68.70%.
- Restored h-only K050 records have all three matched ablations and full
  qualification. Their raw timing pairs use the same RTX5090 session, 64
  inputs, seven passes and three processes as the original cohort. Geometric
  means are recomputed from raw host timings, not the summary median fields.
  Projection MMA bypass versus projection gain has R2=0.941466 for 35 points;
  scalar projection sparsity has R2=0.425498. The historical 30-point values
  remain 0.946/0.413. Attention skipping is slower for all 35.
- At kappa=0.5, four-site Ph has full K050 latency 0.462252 ms and loss
  5.722666, versus seven-site P7's 0.473366 ms and 5.829407, despite much
  less logical sparsity. No seven-site Ph timing is inferred from this.

## Identification and remaining measurements

Each condition has one training seed and all reuse the validation split.
T7/P4 is absent; Ph-to-P7 jointly adds a,m,z,q,k,v and changes equal-tensor
normalization. This cannot identify Q/K/V pressure separately. Geometry is
observed during training, not reconstructed from final performance; it does
not establish a causal mechanism for quality differences.

H-only scaling is untested. Separately prepared Run 033 covers only the five
new seven-site h-only final-K050 benchmarks; its launch and output are outside
this retained-data revision. Local RTX5070 Ti Laptop timing would not match
the RTX5090 protocol. No new run is launched here and no pending result is
written into the paper. Existing signed-density histograms do not cover the
h-only arms; the new site/layer plots use genuine exact-zero counters instead.

## Verification

Four focused tests reproduce the retained reduction, validate scope identities,
count pooling and all five thresholds; test paired arithmetic and loss-pass
sensitivity; verify all 14,240 step records and actual cap flags; and check
the 35 qualified runtime records, matched ratios and plotted coordinates.
Manuscript build, copy hashes and visual checks are recorded in the draft's
17 September pressure-placement review record.
