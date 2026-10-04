# Does corner-tile training lower full-model inference latency?

Question: compare the retained scalar and causal8x16 corner T2/Ph checkpoints at κ=0/.1.
Method: fixed K050-derived h/z consumer with exact trained gates; matched native, no-h/z-skip and sparse graphs.
Coverage: one RTX5090, BF16 B1/T2048, all50304 logits; each of six fresh processes checks338 validation blocks from500 documents.
Each threshold has64 fixed timing inputs, seven randomized passes, three fresh processes and1344 direct pairs; excluded tail1444 tokens.
Caption: left, full-model host geometric mean latency; right, same-input scalar/tile sparse latency ratios. Ratios above1 favor tile.
The no-skip control disables h/z tile bypass and short-row substitution together; remaining components and gates are matched.

At κ=0, tile/scalar sparse speedup is 1.123472x; matched no-skip/sparse ratios are 0.991118x (scalar) and 1.117454x (tile).
At κ=0.1, tile/scalar sparse speedup is 1.111937x; matched no-skip/sparse ratios are 1.066981x (scalar) and 1.192173x (tile).

Caveats: one training seed/GPU; repetitions assess systems variability. Tile checkpoints have higher original validation loss. No quality-matched, decoding, scaling or measured DRAM-traffic claim.
Sources: [table](../results/tables.md), [machine results](../results/results.json), [reducer](../20_analyze.py), [figure script](../21_figure.py), [PDF](../figures/01-corner-tile-latency.pdf).
