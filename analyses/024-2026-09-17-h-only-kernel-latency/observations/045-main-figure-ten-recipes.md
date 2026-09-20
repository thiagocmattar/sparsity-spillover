# Figure22: ten 14M recipes and compact legend

## Question and method

The author requested T1/Ph and the h,z threshold recipe in the 14M main figure,
then explicitly selected all ten recipes from Analysis026 and named the two-site
recipe T2/Ph. The requested final style uses a two-row, five-column legend,
90% of the previous font sizes, and slightly larger panels in the same 8.6 x 3.85 in
canvas. The panels are about 3% wider and 14% taller.

[34_plot_14m_main_figure.py](../34_plot_14m_main_figure.py) preserves every measured
coordinate and adds the retained Analysis026 endpoints. At the author's request,
both panels now omit post-hoc clipping paths and show the 40 unique trained
checkpoints, including Base/ReLU controls. Panel (b) ends at 0.7 ms, giving
the intervention curves more vertical space. T1/Ph is the four-value lambda
sweep; T2/Ph is the four-value kappa sweep. No threshold 0.5 T2 run is invented.
The displayed T2 means gates at h,z; it does not rename the operational A2=m,h
registry entry. The T2 ceiling is 5.3471%.

## Coverage and limits

Loss and sparsity use ordinary final-checkpoint validation over all 338 complete
blocks from 500 documents; 1,444 tail tokens are excluded. Model-wide sparsity is
count-pooled and includes the dense output-head denominator. Latency uses the
retained K050 geometric means on RTX5090, BF16, B1/T2048, full logits, with 1,344
samples per trained checkpoint. Different physical GPU sessions contribute
measurements; small latency differences are descriptive. All 40 trained
checkpoints fit within both panels. Historical clipping measurements remain
in their source artifacts but are omitted from this figure and its data export.

## Caption and manuscript use

[PDF](../figures/22-14m-main-quality-sparsity-latency.pdf), copied to
`manuscript/draft/figures/22-14m-main-quality-sparsity-latency.pdf`, is Figure1 in
`introduction.tex`, label `fig:quality-sparsity-overview`. The caption and displayed
14M cohort count are updated; other author prose is preserved. `main.pdf` is not
rebuilt. The current caption is:

\textbf{(a)} Final validation loss and \textbf{(b)} full-model latency versus $\Smodel$ for the same 40 checkpoints: the Base Model ($T_0/P_0$), GeLU $\rightarrow$ ReLU ($T_1/P_0$), $T_1/P_h$, $T_2/P_h$ with thresholds at $h,z$, and $T_4/T_7$ with no pressure ($P_0$, dash-dot), pressure at $h$ ($P_h$, dashed), or pressure at all thresholded sites ($P_{\mathrm{all}}$, solid). Curves connect separate trained settings: $\lambda=.05,.1,.5,1$ for $T_1/P_h$; $\kappa=0,.01,.05,.1$ for $T_2/P_h$; and $\kappa=0,.01,.05,.1,.5$ for $T_4/T_7$. Vertical dotted lines mark analytic sparsity ceilings. Latencies use RTX5090, batch one and 2,048-token sequences. Measurements span separate GPU sessions, so small latency differences are descriptive.

## Verification

The builder verifies all Analysis026 source hashes, 40 unique checkpoint keys,
all recipe dose grids, matched panels, and bounds. The verifier checks that
post-hoc paths are absent and the latency upper limit is 0.7 ms.
A separate PDF/data check verifies exact coordinate equality to Analysis026,
embedded fonts, page bounds, and identical manuscript copies. The final rendered
page is inspected for overlapping text and legend completeness. Results are in
[data/main-figure-ten-recipes-verification.json](../data/main-figure-ten-recipes-verification.json).

## Result and provenance

All ten recipes are visible in both panels. The two-site series is labeled T2/Ph;
T4/T7 are paired across legend columns for P0, Ph and Pall. No new measurements
or scientific claim is added. Source data and script hashes are retained in
[data/14m-main-quality-sparsity-latency.json](../data/14m-main-quality-sparsity-latency.json).
