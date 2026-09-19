# Figure 17: Two-panel 70M sparsity contribution and latency comparison

PDF: [17-70m-site-groups-sparsity-latency.pdf](../figures/17-70m-site-groups-sparsity-latency.pdf).
Builder: [28_plot_70m_site_groups_latency.py](../28_plot_70m_site_groups_latency.py).
Exact coordinates, pooled counts, source identities and series:
[70m-site-groups-sparsity-latency.json](../data/70m-site-groups-sparsity-latency.json).

## Question and requested scope

Combine the h/z and complementary contribution views of Figures 15 and 16
into one figure for Pythia-70M. Following the user's layout clarification,
panel (a) on the left shows h/z and panel (b) on the right shows the
complement. The panels share a latency scale and recipe legend, with separate
x-axis ranges. Each of 20 trained checkpoints appears once in each panel
at its measured full-model latency, using circular markers throughout.
The 40 displayed markers represent 20 checkpoints, not 40 separate runs.
No training, timing, kernel change or manuscript edit is involved.

## Method and coverage

The recipes are T4/Ph, T4/Pall, T7/Ph and T7/Pall, each with kappa =
0, .01, .05, .1, .5, matching the 70M T/P subset of manuscript Figure 1.
Base model, GeLU -> ReLU and post-hoc clipping are omitted. Recipe colors and
dashed Ph / solid Pall line styles follow Figures 15 and 16. Within each
recipe and site group, lines connect increasing threshold settings.

For each checkpoint, sum integer logical zero-product counts before division:

- h/z: MLP W2 and attention output projection.
- Complement a/m/q/k/v: QKV projection, MLP W1, QK and PV. QK is counted once,
  using the union of zero-operand products from q and k.

Each sum is divided by the same full-model product count, including the dense
vocabulary head, then multiplied by 100. The x unit is contribution to S_model
in percentage points. The two numerators partition all block zero-product
counts exactly, so the two x-values sum to S_model at each checkpoint.
Neither site group is renormalized to its own operation count.

Canonical FP16 logical counts cover all six layers and 338 complete 2048-token
blocks from all 500 MiniPile validation documents: 692224 input tokens, with
a 1444-token excluded tail. Every original logical diagnostic is hash-checked
and reconciled with the retained paper export.

The latency comes from Run035's qualified `k050-70m-v2` kernel: the
shape-specific 70M port of K050. It is BF16 full-model inference on RTX5090,
batch one, 2048 tokens and all 50304 vocabulary logits. Each geometric mean
pools 64 fixed timing inputs, seven passes and three fresh processes
(1344 timings per checkpoint). All settings retain complete-validation
qualification from Run035. The y-axis is not the time spent at the plotted
site group; it is the same measured full-model latency for both markers.

## Caption

**Sparsity contributions and full-model latency on Pythia-70M.** Panel (a)
shows the combined h/z contribution to S_model; panel (b) shows the
complementary a/m/q/k/v contribution, counting QK once. Each checkpoint
appears in both panels at the same full-model latency, with its two x-values
summing to S_model. Colors identify the four T/P recipes, and dashed Ph /
solid Pall lines connect increasing thresholds within each site group.
Panels share the y-axis scale and use independent x-axis ranges.
Counts use the full-model denominator and are pooled over complete validation.
Latencies use the qualified 70M K050 port on RTX5090, BF16, batch one,
2048-token full-sequence inference and full vocabulary output, averaged
geometrically over 1344 timings. The 40 markers represent 20 trained
checkpoints; lines are visual guides rather than fitted relationships.

## Result and limitations

The h/z contribution ranges from 12.74013 to 15.43939 pp; the complementary
contribution ranges from 10.82229 to 25.18031 pp. Full-model latency ranges
from 1.61837 to 3.18577 ms. At kappa=.5, all four recipes have nearly equal
h/z contributions (15.42156--15.43939 pp), while the complementary contribution
spans 13.79668--25.18031 pp. T7/Pall has the largest complementary contribution
but higher latency (1.74842 ms) than the other three high-threshold settings
(1.61837--1.62644 ms).

Both sparsity groups change with the training recipe and threshold; this
comparison does not isolate their individual causal latency effects. Logical
sparsity does not measure matrix-instruction bypass or tile occupancy.
Canonical FP16 counts and BF16 runtime measurements are distinct. One seed
and final checkpoint per training condition are represented. The 70M port
has its own shape-specific implementation and tuning history. No uncertainty
or regression is fitted, and no new manuscript claim is promoted.

## Verification and reproduction

- All 20 source hash/count/coverage checks and exact integer complement
  identities pass; the two contributions also sum to exported S_model.
- All checkpoint identities and latencies equal the 70M Figure 1 subset;
  each latency matches the geometric mean of its three retained process
  means to relative tolerance 1e-12.
- Eight series contain five threshold points each. All 40 coordinates are
  within the linear axis limits: 12.55--15.60 pp for h/z, 10--26 pp for the
  complement, and a shared 1.52--3.28 ms latency range. All checkpoint data
  and plotted coordinates are unchanged from the initial overlay.
- Every existing analysis PDF, including Figures 15 and 16, remains
  byte-identical. The revised one-page PDF was rendered at 1900 pixels and
  visually checked; panel titles, axes and shared recipe legend fit cleanly.
  All fonts are embedded.

```powershell
.venv/Scripts/python.exe -X utf8 analyses/024-2026-09-17-h-only-kernel-latency/28_plot_70m_site_groups_latency.py
```
