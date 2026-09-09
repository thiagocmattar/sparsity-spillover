# O004: Revised displays of existing evidence

These PDFs are owned by Analysis 019 and copied into the active draft. Old run
and analysis figures remain unchanged. Source hashes are in `figures/SOURCES.json`
and `figure-source/SOURCES.json`. The original recipe colors and marker shapes
are retained; new signed-density curves also differ by dash pattern. Preliminary
standalone rendering passed; final compiled-page inspection is recorded in the
draft's revision verification.

| PDF | Question and method | Caption / result | Coverage and caveat |
|---|---|---|---|
| `01-v2-14m-overview.pdf` | Where are low-loss and high-sparsity endpoints? `05_make_figures.py` calls the original coordinate generator, changes its title and adds A0 and quality-cost annotations. | All 30 trained points and the original A0 clipping path; A7-OL1 kappa.5 costs+.6208 loss versus A0 at27.483% sparsity. | Same canonical evaluation as O001. Six of ten clipping points fall inside the original restricted loss axis. Paths connect settings, not Pareto envelopes. |
| `03-scale-transfer-and-reach.pdf` | Which complete-recipe ordering persists, and what changes with the denominator? Original coordinates, revised reach guides and S_block axis. | High-threshold A7-OL1 improves both axes over A4-OL1 at each size; lower-threshold order varies. | All 60 plotted trained/A0-clipped points occur in both rows. Common A7 normalization equals S_block. Same tokens, not tokens per parameter;410M learning rate differs. |
| `05-v4-activation-density-grid.pdf` | Where is the probability mass at zero and away from zero? Rebin the seven retained Run 031 histograms with the original exact native-bin function; verify each zero fraction against the saved density data. | Exact-zero percentages appear above every panel; A4-OL1 has more pooled FFN zeros at kappa.5, but far fewer attention zeros than A7-OL1. | Six panels, A0 repeated, all layers/338 blocks; h:m80:20 and q:k:v equal. FP32 weights/FP16 autocast, no extra clipping. Zero excluded from density; divide by all elements and actual width, no visible-range renormalization; tails retained. Symlog areas are not probability mass. |
| `07-kernel-attribution.pdf` | What produced acceleration? Audited native-relative cohort means and per-checkpoint incremental factors. | Fusion alone accelerates; sparse paths help14/30, GM1.0432; attention-dense is faster 30/30. | O002's BF16 protocol and canonical FP16 horizontal counts. Circle marks c30; OLS is descriptive. Different variants have separately measured native-normalized timings. Restricted axes are explicit. |
| `08-search-history-and-association.pdf` | What happened in the recorded search, and how does selected K050 covary with sparsity? Original retrospective coordinates and fit, with accurate title. | Incumbent reaches 1.7830 at iteration 42; final 30-point native-relative OLS R2=.8167. | Search checkpoint fixed; final cohort varies. Faster later ablation is not substituted into search history. No prospective or causal claim; FP16/BF16 mismatch remains. |
| `sparsification-ladder.pdf` | What nonlinearities, pressure targets and reach define each row? `06_make_recipe_figure.py` compiles retained TeX with revised labels and explicit pressure-target notes. | Original eight recipes and all twelve architecture/topology values retained; reach uses R_arch. | Separately trained rows, T2048; not a training curriculum or quality-constrained bound. |
| `pythia-architecture-sparsification-ladder.pdf` | Where do the recipe sites sit in the graph? Same composite with the updated ladder panel. | Original parallel branches, post-RoPE q/k, pre-output z and all seven labels preserved. | Architecture panel is byte-identical to the original. Same reach and pressure caveats as the ladder. |

The compact 14M operation explanation is a table, from O001 and
`04_make_tables.py`. Its common-denominator projection and QK/PV contributions
add to the measured total. The original three-size operation PDF and all 540
full-range clipping points are retained in the appendix without changes.
