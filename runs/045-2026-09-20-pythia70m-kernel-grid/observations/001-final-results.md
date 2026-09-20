# Verified 70M matched kernel sweep

Question: does frozen Run042 opt073 improve execution of the retained 70M grid
over the original 14M-derived port, with fixed checkpoints and thresholds?

Method: native / original-port / opt073 paired in each of three fresh processes
for all 26 approved checkpoints, one physical RTX5090, batch1, sequence2048,
all50304 logits. All338 validation blocks qualify; 64 inputs x7 randomized
passes x3 processes give1344 host timings per implementation/checkpoint.
Geometric means are primary; no selection of the best backend per recipe.

Result: all78 processes qualify. Native Base is1.611048ms. The original port's
best recipe (T7/Ph kappa=.5) takes1.589621ms, just1.013479x native Base. The
optimized path takes1.088146ms (1.480544x). Every checkpoint improves over its
own original port, but only six optimized endpoints beat native Base. Low-loss
T2/Ph endpoints remain slower than native Base. Quality and logical counts are
retained separately from BF16 qualification diagnostics.

Caveats: dense optimization also contributes; there is no pure skip-toggle
attribution for this optimized 70M grid. Sparse exploitation does not imply a
scale-dependent speedup trend or a quality-matched advantage. The separate
Run046 kappa=.5 endpoint was not measured with opt073.

Evidence: `results/matched-grid.json`, `results/local-verification.json`, raw
`artifacts/`, `transfer/inventory-tail001.json`, `prelaunch/teardown-final.json`.
All819 returned files verified; both owned Pods deleted. Analysis028 owns the
publication figures, captions and source-generating scripts. Run scripts
`02_benchmark.py` and `06_reduce.py` generate and reduce the measurements.
