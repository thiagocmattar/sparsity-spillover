# Methodology source audit: notation, provenance, and presentation

Independent source audit, 2026-09-05. No manuscript or scientific-code edits. Read the original `manuscript/methodology.tex`, current draft wrapper and bibliography, the ladder/map TeX and companion Markdown, operational definitions/methods/metrics/data contracts, and Analysis 013's evidence scope. This is preparation for two scored draft reviews, not a score of a methodology draft that has not yet been written.

The forthcoming rubric uses five equally weighted criteria: citation/definition provenance; main/appendix progression; notation/figure consistency; workload/systems interpretation; reader utility/economy. Each uses 1 = blocking, 2 = major revision, 3 = competent but uncompetitive, 4 = strong, 5 = exceptional; maximum 25.

## Recommended notation

Use `\mathcal{S}_{\mathrm{model}}`, `\mathcal{S}_{\mathrm{block}}`, and `\mathcal{S}_{\mathrm{model}}^{\max}`. Calligraphic S makes the sparsity family recognizable while remaining distinct from the attention score tensor S in the old forward equations. Define these as fractions; render percentages only in displays explicitly labeled with `%`. Do not multiply by 100 in some equations but call the same quantity a fraction elsewhere.

The old methodology also uses `\mathcal S` for a set of diagnostic sites. Reserve that glyph for the metric in the new draft and use a separate symbol for gate sites and pressure sites, for example `\mathcal G` and `\mathcal P`, with topology denoted by tau. Keep the historical `R_*` fields in immutable artifacts. One provenance note can map the paper notation to those fields; raw artifact keys are unnecessary in the paper's main text.

Most important semantic point: the superscript max is **selected-site all-zero reach**, not a universal upper bound on all observed model zeros. Observed sparsity includes natural zeros outside that selected reach. A0 consequently has zero selected-site reach but can have nonzero measured sparsity. Do not write `S_model <= S_model^max`, label their ratio a bounded utilization, or imply the difference is an ordinary unexploited-capacity measure. These restrictions follow directly from `research/DEFINITIONS.md` and `research/METRICS.md`.

Use “scalar multiplication” or “scalar product contribution” for the atomic counting unit. “Scalar product” commonly means a dot product; the old source uses it to mean one multiplication and risks confusing the unit.

## Main text needed to interpret the experiment

The main methodology can be short while retaining the information that changes scientific interpretation:

- Random initialization from Pythia architecture configurations, rather than released weights; paired conditions match initialization, realized data order, and token budget. Name MiniPile and the three architecture sizes without a full optimizer table.
- Distinct gate and pressure site sets. The ladder is a map of alternatives, not proof of a fully crossed experiment or of every row at every size. A4-OL1 versus A7-OL1 changes pressure sites as well as gates.
- The two gate equations and the meaning of kappa. Equality survives; one-sided zero-threshold gating is ReLU, while symmetric zero-threshold gating is identity. The gates remove values; they do not subtract the threshold from survivors.
- A one-equation pressure objective, stating that it averages per-tensor mean absolute activations equally across captured site/layer tensors. Element-count weighting would define a different objective.
- A concise distinction between naive L1 and OL1. OL1 uses task-only AdamW moments, a conflict-conditioned correction, and a trust cap; it does not ensure loss preservation or remove all pressure hyperparameters.
- One definition of observed product sparsity, the dense-head denominator, and the selected-site reach ceiling. State full-sequence uncached causal evaluation and exact-zero counting rather than treating the metric as inference latency.
- Validation coverage and pairing of quality and sparsity: all 500 documents, 338 complete length-2048 blocks, excluded 1444-token tail, same checkpoint and evaluation configuration. One-seed and selected-size comparisons are descriptive.

The appendix should carry the full graph, site table, exact threshold/backward semantics, operation-wise counters, integer-pooling rule, ceiling derivation, and complete OL1 procedure. Place architecture dimensions, immutable data/config identities, global batch, learning-rate/budget grids, and coverage tables there. Avoid copying the original source's entire topology registry or 12B example into this study's main text.

## Source issues that must not be copied blindly

The old methodology is headed as aligned to a historical upstream commit. Current operational methods are the applicable contract. Its raw-task-gradient equations omit task clipping in OL1, and its trust budget is optional; the implemented protocol clips the task gradient before AdamW and requires a positive OL1 budget. Naive L1 clips the combined gradient. These differences were already acknowledged in `research/MANUSCRIPT.md`; they need correct new prose, not a repeated research-design dispute.

The original file embeds `\appendix` and page breaks. Importing it wholesale into the current wrapper would duplicate document structure and overwhelm the compact main section. Reuse its mathematics selectively in separate main and appendix files.

There is a stale operational-status discrepancy: `research/MANUSCRIPT.md` still says 410M remains unobserved, whereas `research/INDEX.md` and completed run/analysis records describe Run 019 and subsequent 410M systems work. The new draft should rely on verified run/analysis scope rather than that stale summary. No evidence promotion is implied by correcting the description.

## Figure assessment

The existing ladder already reports all-zero reach for 14M, 70M, and 410M. Its stated tuples are `(L,d,V)=(6,128,50304)`, `(6,512,50304)`, and `(24,1024,50304)` at T=2048 and FFN width 4d. A direct arithmetic check reproduces the displayed A1-H/A4/A7 values after rounding: 4.3/12.8/30.0%, 12.4/37.1/49.4%, and 24.9/74.8/87.2%. No new measurement was performed.

A second plot of the same ceiling columns adds little. Retain the ladder, update its metric glyph consistently, and explain how its ceiling is computed. A cross-family plot would need declared architecture configurations and a new analytic comparison; no such comparison should be invented. If an additional explanatory chart is eventually wanted, decomposing the existing three Pythia denominators into head, projections, and attention would add information more effectively than repeating the ceiling table. It is optional, not needed for this compact methods section.

The ladder companion Markdown incorrectly says every row retains the preceding row's configuration. Some rows replace L1 with OL1 or remove pressure when expanding the gate topology. The paper caption should say the rows map intervention alternatives. Preserve the distinction between the architecture diagram's site p and active gate sites: p is an observed attention-probability operand, not one of the ladder's seven gated sites.

Caption essentials: identify upper graph/lower ladder, define compact q/k as post-RoPE and z as context before the output projection, state analytic reach at T=2048 with the dense head retained, and distinguish candidate rows from evaluated coverage. Explain pressure targets once in a compact note or appendix table. Avoid turning the caption into the optimizer protocol. Figure numbering and placement should introduce the ladder near the first methods discussion rather than leaving it as a detached closing exhibit.

The existing ladder uses color-only dots for gate families. Source inspection identifies a possible grayscale/accessibility weakness, but this audit did not perform a new rendered visual inspection. A redundant symbol or short code would help if it can be added without crowding. The compiled draft should be inspected at its actual reading width because the composite's source width is substantially larger than the text column.

## Citation and workload alignment

The current bibliography already contains Pythia, MiniPile, TEAL, the sparsification methods, and optional gradient-surgery context. Cite the architecture and dataset when introducing the setup. Exact local initialization, tokenizer, and training recipes need local provenance; they do not inherit the released Pythia pretraining recipe from the architecture citation. The original [Pythia paper](https://arxiv.org/abs/2304.01373) describes a released suite initially starting at 70M; the 14M shape should therefore be tied to the pinned configuration rather than inferred from that paper alone.

The bibliography currently lacks AdamW. If the appendix develops its update, add the original [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) citation (ICLR 2019, verified). PCGrad/Bloop may contextualize a projection step in the appendix but do not constitute evidence that this implementation has their guarantees. Do not restore a gradient-specific related-work subsection.

The product denominator omits softmax, normalization, residual additions, selection, packing, and other execution overheads. It is valid as a declared logical-multiplication estimand, not as total FLOPs, total compute saved, or measured speedup. Keep full-sequence forward timing distinct from autoregressive decoding: the former covers the method's declared sequence workload, while decoding introduces cache, shape, and memory-traffic differences. TEAL's projection-based reported sparsity is not numerically interchangeable with this full-forward metric. Methods can state the intended timing contract without claiming a new positive runtime result or conflating the pending agentic work with a completed experiment.
