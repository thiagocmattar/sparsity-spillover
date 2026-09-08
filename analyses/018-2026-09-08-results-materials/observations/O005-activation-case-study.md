# Activation mass at manuscript sites

## Question

How can similar FFN hidden zeros coexist with different attention opportunities?

## Method

Pool integer counts across layers before division. Restore one column per site h,m,q,k,v and one row per κ=0,.05,.5. Show four count-derived magnitude-bin masses with one shared recipe legend. The same A0 baseline appears in all three rows. RMS and all retained bands remain in the table.

## Coverage

Seven checkpoints: 14M A0 plus A4-OL1/A7-OL1 at κ=0,.05,.5. Five common sites from complete activation-diagnostic passes. q/k are post-RoPE as in the manuscript ladder. Unless the section specifies runtime timing inputs, validation covers all 500 documents packed into 338 complete 2048-token blocks, excluding the 1,444-token tail.

## Figure and caption

[Publication PDF](../figures/05-activation-mass-grid.pdf)

14M activation magnitude distributions. Columns identify sites; rows show κ=0,.05,.5. Each panel gives probability mass in the retained bins {0}, (0,.001], (.001,.01] and >.01. The y axis is logarithmic above .01% and linear around zero; exact empty bins remain visible. The single shared legend identifies A0, A4-OL1 and A7-OL1. Site names follow the manuscript ladder: h is FFN hidden, m FFN input, q/k post-RoPE query/key and v value. Lines guide comparisons across categorical bins and are not a smooth density estimate. All panels share their scales and one legend.

## Result

At κ=.5 both recipes exceed 99.8% h zeros. A4 query/key mass below magnitude .01 is 47.64%/56.12% with fewer than .001% exact zeros, while A7 query/key exact zeros are 93.54%/94.54%; v exact zeros reach 98.71%. A4 query RMS (.146) is lower than A7 (.483), despite fewer zeros.

## Caveats

These are coarse magnitude-bin distributions, not signed densities. The requested finer bins at .05/.5 cannot be recovered from the stored >.01 tail; their measurement is pending the design in ACTIVATION-DIAGNOSTIC.md. No extra bin is fabricated. A0 lacks a/z near-zero statistics. The post-Wo attention_output diagnostic is omitted, not relabeled as pre-Wo z. Separate activation and logical passes can differ numerically. Marginal distributions alone do not identify a causal spillover mechanism.

## Source script and evidence

`plots.py:activations`, invoked by `01_build.py`. Supporting evidence: tables/activation-statistics.md; tables/activation-case-mass-bins.md; manuscript/artifacts/pythia-architecture-sparsification-ladder.pdf.
All source identities, integer counts and exact values are retained in [figure_data.json](../figure_data.json).
