# 001: Matched quality panels with conditional site savings

**Question.** How does a three-panel composition connect the matched 14M/70M
quality-sparsity trade-offs to Table 2's operation-level saved-time evidence?

**Method and coverage.** Select the intersection of the two existing trained
recipe sets, yielding seven recipes and 27 unique checkpoints per size. The
five pressure recipes each cover kappa=0,.01,.05,.1,.5. Preserve ordinary final
validation losses and integer-pooled logical-product sparsity from Analyses027
and029, and the 20 retained Base/ReLU post-hoc settings per size. Full validation
has 338 complete 2,048-token blocks from 500 MiniPile documents, excluding the
1,444-token tail. Only common recipes receive legend entries.

The six middle bars reproduce Analysis024's Table 2 export from Run037. For its
14M T7/Pall, kappa=0.5 checkpoint, saved time is full-model latency with one
skipping path disabled minus latency with all paths enabled. Timing uses RTX5090,
BF16, batch one, 2,048 tokens and the full vocabulary output. Every mode has
three independent processes, each with 64 timing inputs and seven passes.
All 30 processes qualified; the upstream audit checked 26,880 raw timing samples.
Whiskers are the extrema of all off/on process-mean differences, not confidence
intervals. QK jointly controls q/k; PV controls v.

**Result.** The figure shows the same seven styles at both scales with one
legend. The center chart preserves the large positive h/z effects and small
negative effects elsewhere, on an unbroken linear axis:

| Site/control | Saved time (microseconds) | Process span (microseconds) |
|---|---:|---:|
| a / QKV | -1.6 | [-6.0, +3.8] |
| m / FFN-up | -2.4 | [-5.1, +2.7] |
| h / FFN-down | +150.1 | [+145.4, +154.4] |
| z / attention output | +29.0 | [+23.7, +35.7] |
| q,k / QK | -2.9 | [-7.3, +1.1] |
| v / PV | -6.2 | [-9.0, -2.2] |

**Legend and caption.** (a) Pythia-14M and (c) Pythia-70M quality-sparsity
trade-offs for the 27 trained checkpoints shared by recipe at both sizes.
Colors and line styles identify Base, GeLU-to-ReLU, T2/Ph, T4/Ph, T7/Ph,
T4/Pall and T7/Pall; curves connect the five threshold settings. Dotted control
curves show retained post-hoc clipping; high-loss tails exceed the focused view.
The approved revision omits vertical ceiling guides and the graphic's footer.
(b) Conditional saved time by skipping control for 14M T7/Pall at kappa=0.5.
Positive bars mean reduced full-model latency. Whiskers show process extrema;
the bars are not additive allocations of total runtime savings.

**Caveats.** Each training recipe has one seed. The side panels use different
axis ranges and share recipe identities, not checkpoint weights. Logical
sparsity is not a runtime metric. The center panel measures one 14M checkpoint
and implementation; it is not a decomposition of the 70M panel or of every
recipe in (a). Its process ranges are descriptive and may cross zero.

**Sources and verification.** `../01_figure.py` generates
`../figures/01-quality-sites-quality.pdf`; `../02_verify.py` checks the exact
cohort, table values, source hashes, PDF bounds and fonts. The exported evidence
and verification are `../data/figure-data.json` and `../data/verification.json`.
The rendered one-page PDF was visually checked for labels, whitespace, bar
extrema, shared legend and clipping. The approved figure replaces
`fig:quality-sparsity-overview` in the introduction as
`manuscript/draft/figures/24-quality-sites-quality.pdf`. Its caption explains the
three panels along the author's revised argument: broad placement trade-offs,
conditional h/z benefits motivating T2/Ph, and the 70M quality comparison.
It retains the non-additive interpretation and process-span definition removed
from the graphic, and qualifies the near-baseline T2/Ph claim to kappa<=0.1.
The 21-page manuscript build has resolved references and no overfull boxes;
the figure/caption on page 2 were visually checked. Surrounding author prose
was preserved byte-for-byte during replacement.
