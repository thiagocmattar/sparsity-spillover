# Where do the corner-tile endpoints sit in the previous quality-latency frontier?

Question: place the two new 14M T2/Ph corner endpoints at κ=0/.1 in the
manuscript's previous quality-on-X, latency-on-Y comparison and demarcate the
observed frontier. The user requested this figure in Analysis041.

Sources: the current manuscript asset
manuscript/draft/figures/02-14m-70m-latency-quality.pdf is a byte-identical copy
of Analysis038's scale figure. Its 36 execution points from 33 checkpoints
cover Base (kernel/PyTorch), T2/Ph and T7/Ph across 14M/31M/70M. They are
retained unchanged. Add exactly two tile endpoints from Run060/Run061;
the two Run058 scalar checkpoints measured in Run061 are auxiliary references
shown only in the 14M detail, excluded from frontier membership.

Method: X uses ordinary canonical FP16 final-checkpoint loss from the original
complete-validation evidence, matching the historical figure. Y uses Run061
sparse synchronized-host geometric mean latency; each threshold has 64 inputs,
seven randomized passes and three fresh processes, 1344 pairs. Whiskers span
the minimum/maximum process geometric means, not confidence intervals.
Checkpoint identities link each timing point to its original quality source.
Every Run061 graph passed own-eager BF16 numerical/loss qualification.
Verify source SHA256, all 120 transferred files, all 35 original quality-source
identities and selected raw timing/quality data before plotting.

Coverage: all 500 MiniPile validation documents, 338 complete 2048-token blocks,
692224 input tokens and 691886 predictions, excluding 1444 tail tokens.
One training seed, 1234; BF16 latency on RTX5090, B1/T2048, uncached, all 50304 logits.
Different scales use architecture-specific kernels and different timing sessions.

Caption: (a) the previous cross-scale comparison with two filled teal diamonds
for 14M corner tiles. Blue dashed/orange solid curves retain historical T2/Ph
and T7/Ph. Circles/triangles/squares identify 14M/31M/70M. Hollow/filled gray
markers identify kernel/native Base. Both overview axes are logarithmic.
(b) a labeled linear 14M detail additionally shows hollow teal diamonds for
the matched Run061 scalar references. Teal callouts mark tile thresholds and
a blue callout marks historical scalar κ=.5. The gray dotted staircase traces
nondominated observed endpoints among the 36 historical points plus two tiles;
it is neither a fitted curve nor interpolation of achievable models.
Smaller values on both axes are preferred.

Results:

| Endpoint | Canonical FP16 loss | Sparse latency (ms) | Process range (ms) |
|---|---:|---:|---|
| Tile κ=0 | 5.553770 | .540220 | .539432-.541275 |
| Tile κ=.1 | 5.596746 | .508340 | .506742-.510481 |
| Historical scalar T2/Ph κ=.5 | 5.536255 | .506323 | Separate retained session |

Neither tile endpoint expands the observed frontier. The 15-point historical
Pareto set remains unchanged. Historical scalar T2/Ph κ=.5 has lower loss
and lower reported latency than both tiles: the latency gap is .033897 ms
against tile κ=0, but only .002016 ms (approximately .4%) against tile κ=.1.
The earlier within-Run061 result still holds: tiles reduce latency 10.990%/
10.067% versus their matched scalar κ=0/.1 controls, with higher loss.
These answer different comparisons.

Caveats: Run061's h/z gate-fused kernel and historical kernel variants/sessions
differ. Raw coordinates are neither pooled nor normalized across sessions.
The frontier and dominance are descriptive arithmetic on retained measurements,
not evidence of a significant advantage for the small .002 ms gap, or a controlled
tile-versus-scalar κ=.5 experiment. Canonical FP16 quality is distinct from the
BF16 own-eager loss used to qualify the latency kernels. There is no equal-quality,
decoding, training-speed, portability or fitted scaling-law claim. No new
measurement, cloud resource, finding promotion or manuscript edit is performed.

Artifacts: [PDF](../figures/04-quality-latency-frontier-with-corner-tiles.pdf),
[exact coordinates, frontier and input hashes](../data/quality-latency-frontier.json),
[table](../results/quality-latency-frontier.md),
[generating script](../02_quality_latency_frontier.py).
The previous three Analysis041 PDFs and the source manuscript figure are
preserved byte-for-byte.
