# Proposed finer activation-distribution measurement

Status: design prepared for user review; no diagnostic implementation or launch.

The requested finer bins cannot be recovered from the saved cumulative counts
at 0, .001 and .01. The final checkpoints are present locally. This proposal
adds a measurement of those checkpoints; it does not retrain them.

## Question and comparison

How does activation magnitude mass change between A0 and A4-OL1/A7-OL1 as
the trained threshold increases from 0 through .05 to .5? The middle value
.05 is an existing treatment, selected to show an intermediate response.

Use seven 14M step-712 checkpoints: A0 from Run 004 and A4-OL1/A7-OL1
from Runs 015/014 at kappa=0,.05,.5. These originated from matched random
initialization and data-order seed 1234, 712 AdamW updates and 1,493,172,224
input tokens. Loading their final weights is post-hoc evaluation, not new
random initialization or continuation training. No optimizer step is performed.

Preserve each saved topology/gate configuration: A0 stock GELU; A4 one-sided
G+ at a,m,h,z; A7 additionally symmetric Gpm at post-RoPE q,k and v. Both
OL1 recipes were trained with pressure at their gated sites. Pressure is
inactive during evaluation. No gate, weight, clipping rule or threshold changes.

## Measurements and coverage

Use FP32 weights with FP16 autocast, eager attention and the same pinned
MiniPile validation cache. Evaluate all 500 documents packed into 338 complete
2048-token blocks: 692,224 inputs, 691,886 prediction targets, excluded tail
1,444 tokens. Recompute paired validation loss as an execution check.

Capture all seven manuscript sites h,m,a,z,q,k,v after their configured gates;
q/k are post-RoPE and z is immediately before Wo. This also supplies the
previously missing A0 a/z distribution counts. Pool integer counts across
layers and blocks before division.

Proposed magnitude-bin boundaries: exact zero separately, then .001, .01,
.05, .1, .5, 1 and 5, plus the upper tail. Right-closed nonzero intervals
preserve the saved cumulative-threshold convention. This creates nine mass
bins, including explicit boundaries at both nonzero plotted kappa values.
Gate equality survives; exact boundary values must be counted correctly.

Retain per-site/layer integer histogram and exact/near-zero counts, total and
nonfinite counts, sum/squared sum/absolute sum, RMS/L2, coverage, checkpoint and
validation hashes, gates and environment identity. Original checkpoints stay
unchanged. Weight norms and logical-product counts already exist; they are not
needed to redraw these marginals. Gradient conflict/OL1 training-boundary
metrics cannot be reconstructed here. No raw activation archive is proposed.

## Interpretation and verification

The manuscript activation case (Analysis 018 O005) gains measured magnitude
distributions. A shift from small nonzero mass to exact zeros at q/k/v would
support its descriptive gate-placement explanation; its absence or a contrary
intermediate response would limit that explanation. This is not a causal
isolation of pressure from gates, because the complete recipes differ in both.
No smooth signed density or seed uncertainty is inferred from magnitude bins.

Check histogram partitioning, finite counts, exact boundary behavior, hook
placement, full validation coverage and checkpoint gate round-trip identity.
Compare the original thresholds/RMS/loss against the retained passes and
report numerical differences rather than silently mixing incompatible passes.

## Execution proposal and confirmation boundary

All seven 56.3-MB checkpoint files are local. The laptop GPU is an RTX 5070 Ti
with 12 GiB VRAM; the read-only inventory found about 3.2 GiB in use. No cloud
resource is proposed. After design confirmation, implement the next numbered
diagnostic run, run focused/full required checks and a bounded resource smoke,
then report measured headroom, ETC, monitoring and artifact inventory for
launch confirmation. No full measurement has been authorized or started here.

Repository AGENTS.md requires design confirmation before experiment code and
explicit launch approval after implementation. Additional retained measurements
can be requested during design review before launch.
