# O003: cross-size intervention and ReLU/clipping comparisons

Question: do complete-recipe relationships recur, and how does ReLU affect later clipping?
Method: identity-checked FP16 join, measured p=0 increments, exhaustive observed comparisons.
Coverage: 36 canonical + 360 clipped records; 96 distinct evaluations in the
main/full cross-size figures (36 trained + 60 A0/A1-H clipped), repeated in two rows.
Legend/caption: filled endpoints, open clipping points; common A7 denominator
for block normalization; actual A4 reach for A1-H plus clipping. Main panels
report all omitted high-loss points; full view retains them.
Result: high-threshold A7-OL1 improves both axes over A4-OL1 at all three sizes.
ReLU has a recurring favorable intermediate clipping region, with extreme-target
reversals at 14M/410M. Exact numbers, run IDs and mismatches are in
manuscript/draft/revision-v2/relu_clipping_findings.md and data/cross-size-audit.json.
Caveats: one seed, unequal tokens per parameter, lower 410M LR; no interpolation,
quality scaling law, or new measurement. Absolute and same-size-relative losses differ.
Source scripts: ../02_cross_size.py and ../03_figures.py.
