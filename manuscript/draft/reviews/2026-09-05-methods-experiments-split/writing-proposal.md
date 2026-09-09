# Experimental Study: independent writing proposal

Prepared from the current `methodology.tex` and `main.tex`. This proposal is unscored and does not edit active manuscript files. It instantiates the existing intervention/measurement definitions for the existing study; it does not introduce an experiment, schedule, or finding.

## Proposed opening

We examine how the additional effect of activation pressure depends on thresholding and placement, and whether the resulting account remains informative across model sizes. We use randomly initialized Pythia-family models with 14M, 70M, and 410M parameters, trained and evaluated on MiniPile. Their shared parallel attention and FFN architecture gives the interventions a consistent interpretation across sizes. Detailed contrasts at 14M examine individual and joint changes; selected larger-model comparisons probe transfer. Validation cross-entropy and model-wide sparsity are paired at the same checkpoint and forward configuration.

Figure~\ref{fig:intervention-ladder} combines the Pythia block with the intervention ladder. The sites are the normalized attention and FFN inputs ($a,m$), FFN hidden activation ($h$), concatenated attention context ($z$), and query, key, and value operands ($q,k,v$). Query and key gates follow RoPE, and the context gate precedes the attention-output projection. A0 retains the stock model; A1-H replaces GELU at $h$ with ReLU. A4 applies one-sided thresholding at $\{a,m,h,z\}$, and A7 additionally applies symmetric thresholding at $\{q,k,v\}$. Pressure variants target all gated sites in their corresponding ladder row. The ladder specifies these alternatives and their architectural reach; the evaluated subset is stated with each comparison.

Within a model size, matched contrasts share initialization, data order, and training budget. Adding pressure with gates fixed tests its additional benefit or cost for that placement. Comparing A4 with A7 without pressure tests the broader gate set; comparing A4-OL1 with A7-OL1 changes both gate and pressure sets and therefore compares complete recipes. These comparisons distinguish controlled individual changes from joint interventions. Across sizes, changes in observed sparsity must also be read alongside changes in architectural reach: identical gate placement can expose different shares of the multiplication workload. This separates the question of how learning responds from how much computation that response can affect.

## Structural notes

- Keep this as three paragraphs, optionally under a single “Experimental Study” heading. A second layer of short setup/ladder/comparison headings would add little at this length.
- Move Pythia/MiniPile, A* labels, the named site map, and within-size matching out of Methodology. The combined figure belongs here, after the recipe paragraph or the complete opening, with its compact explanatory caption.
- Methodology should retain the generic gate/pressure choices, gate formulas, pressure objective and concise OL1 idea, observed sparsity, and selected-site reach. Specific sites in an example can be replaced by “an FFN hidden activation” or “a value operand” if an intuitive reach example is retained there.
- The setup/recipe paragraph can be shortened if its site list duplicates a fully legible figure. The comparison-interpretation paragraph is the higher-priority text to preserve: it tells a reader what an eventual result establishes.
- Coverage, selected cohorts, and matching must follow the actual reported results. The existing study supports detailed 14M contrasts and selected 70M/410M extensions; the wording deliberately avoids implying a complete factorial experiment at all sizes.
- The larger-scale purpose should remain an explanatory transfer test. Do not label it a scaling law or assume that a higher aggregate sparsity fraction demonstrates a stronger learned effect.

## Review rubric for the actual draft

Five equal 1–5 criteria: clarity/precision; section separation; question-driven progression; economy; reader payoff. Anchors: 1 = blocking; 2 = major revision; 3 = competent; 4 = strong; 5 = exceptional. The proposal itself is not assigned a score. The first and second actual-draft reviews will each report /25 with concrete evidence and prioritized revisions.
