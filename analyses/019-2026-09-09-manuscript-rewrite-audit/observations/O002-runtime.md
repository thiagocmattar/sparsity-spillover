# O002: Native speedup, sparse-path attribution and vector sparsity

Question: How much of the recorded acceleration survives comparison with
matched fusion, and what do the retained BF16 row diagnostics establish?

Method/source: `02_audit_runtime.py` uses read-only Run 029 reducer functions,
verifies3542 original source artifacts by bytes/hash and recomputes201600 paired
timings across five candidates,30 canonical checkpoints and three processes.
It rechecks full-validation qualification, process medians and geometric means.
It separately reads each K050 replicate1 untimed diagnostic and reconciles
row-NNZ integer histograms with actual-operand site counts. Outputs and source
identities are in `runtime-audit.json`; `04_make_tables.py` and
`05_make_figures.py` format the derived displays.

Coverage: RTX5090, BF16, batch1, uncached2048-token sequences with50304 logits
per input position. Qualification covers all 338 validation blocks. Timing uses
64 fixed inputs, seven randomized paired passes, three processes:1344 paired
host ratios per checkpoint/candidate. Compile, graph capture, static weight
preparation and equal input staging are excluded; activation-dependent work,
embeddings, blocks, final normalization and full vocabulary logits are included.

Results: native-relative checkpoint geometric means are1.233991707 for K050,
1.182857977 for all-skips-off and1.250558753 for attention-dense. K050 divided
by all-skips-off gives1.043228968, helping14/30. Attention-dense is faster 30/30.
P0 qualifies4/30; its subset mean is not a full-cohort comparator. At c30,
K050's actual latency summary is.473149ms, all-skips-off.620963ms and
attention-dense.467858ms; K050's paired native summary is.844124ms.

The canonical-FP16 sparsity association has R2=.816717670 for native-relative
speedup and.503696247 for incremental sparse-path factor. Removing c30 gives
.852125575/.407962253; removing A7-OL1 gives.852004954/.491027887.
Every leave-family and eligible within-family fit is retained. These are
descriptive OLS fits with an intercept and equal checkpoint weights.

BF16 all-zero row fractions at kappa.5: A4-OL1 h83.7242%, z97.1540%; A7-OL1
h63.1242%, z91.2026%. Each of six layers has692224 rows; h/z widths512/128.
The per-layer tables retain means and all-zero rates. At c30, attention
instrumentation pools QK issued/skipped125611580/173429188 and PV
98060121/200980647:57.9952% and67.2084% bypassed, with net attention overhead.

Caption/legend: runtime figures distinguish native-relative GM from the ratio
of separately measured native-normalized speedups. Absolute milliseconds are
GM of three process medians, each over448 timings; their ratio need not equal
the GM of raw paired ratios. Historical search1.7830 and final-sweep1.7832 are
distinct measurements; the1.8022 maximum belongs to attention-dense. Row tables
describe actual projection inputs, separately from FP16 pooled h/m histograms.

Caveats: one human-guided development trajectory, one GPU/workload and no
independent training replications. The default agent name does not verify each
served model. FP16 counts and BF16 timings remain different measurements.
Retained BF16 scalar counts exclude probability underflow and are only a lower
bound. Row histograms lack token identities, coordinate selection and context
perturbations. Biases can survive zero projection inputs; no branch-collapse
or conditional-computation claim follows. Either complete operand fragment
zero permits instruction skipping; zeros in both fragments are not required.
