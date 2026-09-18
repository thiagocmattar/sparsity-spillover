# Appendix A3: Full-model speedup relative to the dense reference

PDF: [A3-dense-reference-speedup.pdf](../figures/A3-dense-reference-speedup.pdf).
Manuscript placement: [Results appendix](../../../manuscript/draft/results-appendix.tex),
supporting [Section 4.5](../../../manuscript/draft/kernel-autoresearch.tex), `sec:kernel-autoresearch`.

## Question, method, and coverage

What does normalizing candidate latency to each size's dense candidate show?
For every one of the 32/22 primary checkpoints compute
t_candidate,A0,m / t_candidate,recipe,m. The common dense numerators are
0.651573035007824 ms for 14M K050 (Run029) and 3.313247016348893 ms for
70M k050-70m-v2 (Run035). The recipe's measured optimized latency is the
denominator. All values are computed from unrounded geometric-mean host times.
Figure 5's [observation](017-quality-latency-caption.md) defines the workload,
qualification, ordinary-final loss, and canonical sparsity sources.

## Publication caption

**Full-model speedup relative to the dense reference.** Panels show 14M and 70M,
with 32 and 22 qualified checkpoints. The Y axis is t_A0,m/t_recipe,m using the
optimized full-model candidate for both terms: K050 at 14M and k050-70m-v2 at 70M.
The dense-reference latencies are 0.651573 and 3.313247 ms, respectively, and
the horizontal line marks 1×. This is a dense-reference deployment comparison,
not the native/optimized speedup of the same recipe. Measurements use RTX 5090,
BF16, batch one, 2,048-token full-sequence inference with all 50,304 logits;
each time is a geometric mean over 64 inputs × seven passes × three processes.
Canonical logical sparsity is measured in FP16 over complete validation.
The 14M dense reference and all models except 7-site OL1(h) are from Run029;
the latter are from Run033 on a different GPU/host. All 70M values are from
Run035. A0 normalization does not remove session differences. Recipe lines
connect separately trained settings in increasing κ, not training trajectories
or attainable interpolations; labels identify .05 and .5. No training-seed
uncertainty is represented.

## Result and proposed manuscript writing

> Normalization to the optimized dense model permits a within-size deployment
> comparison without using each recipe's different native execution time.
> At κ=.5, 14M 4-site OL1(all) reaches 1.418× against dense K050, while
> 7-site OL1(all) reaches 1.376× despite greater logical sparsity. At 70M,
> 7-site OL1(h) and OL1(all) reach 2.047× and 1.895×, respectively. These
> speedups describe different trained models and quality costs, so they must
> be interpreted together with the quality-latency panels in Figure 5.

## Caveats and provenance

The model sizes use distinct shape-specific implementations. Neither equal
optimization effort across sizes nor an isolated attention-skipping benefit is
established. Reference and recipe times can be from different sessions at 14M.

Source: [paper_execution_figures.py](../paper_execution_figures.py),
[13_rebuild_paper_figures.py](../13_rebuild_paper_figures.py),
[exact reference identities and times](../data/paper-derived.json), and
[unrounded checkpoint rows](../data/paper-checkpoints.json).
The proposed paragraph remains separate from the manuscript TeX.
