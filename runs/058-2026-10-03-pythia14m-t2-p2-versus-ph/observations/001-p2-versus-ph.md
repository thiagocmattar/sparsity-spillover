# T2/P2 versus T2/Ph

Question: does adding z to OL1 pressure improve the same h/z gate recipe?

Method: fresh matched random14M runs, seed1234,712 updates, kappa0/.1, h/z unilateral gates; Ph targets h, P2 targets h/z. Pool integer counts across all6 layers before dividing.

Coverage: all500 MiniPile validation documents,338 complete2048-token blocks,692224 input tokens, excluded1444-token tail. All checkpoint and diagnostic source identities are linked in results/results.json.

Result: At kappa=0, P2 raises loss versus Ph by +0.153062; logical opportunity changes by +0.3257 percentage points. At kappa=0.1, P2 raises loss versus Ph by +0.178198; logical opportunity changes by +0.0262 percentage points.

Figure01: Ph blue and P2 orange; final full-validation cross-entropy (lower is better) and count-pooled logical-product opportunity (higher means more logical zeros). Lines connect evaluated thresholds; intermediate values were not measured.

Figure02: each site shows zero-element percentages (solid bars) and fully empty8x16 tiles (hatched bars). Tiles stay within sequences, partition all values without padding, and require all128 entries to be exactly zero. Both use count-first pooling.

Caveats: one seed, two thresholds, equal cutoff does not match sparsity. Pressure averages site/layer tensor means, so adding z changes the relative h contribution. Empty tiles and logical opportunity are not measured speedup. No tile-pressure intervention or manuscript change.

Source: ../21_analyze.py; ../results/results.json and ../results/tables.md; ../figures/01-paired-quality-opportunity.pdf and ../figures/02-site-zeros-tiles.pdf.
