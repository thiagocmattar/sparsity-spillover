# 14M main figure and 70M scale check

## Question and approved scope

The author rejects the previous twofold-speedup scale narrative and requests
a 14M-only Figure 1. The introduction should treat 70M as a scale check and
state that useful skipping occurs at both sizes, while a sparse implementation
requires adaptation. Changes to other sections will follow separate author
guidelines. The author explicitly requests no recompilation of `main.pdf`.

## Method, coverage and provenance

[34_plot_14m_main_figure.py](../34_plot_14m_main_figure.py) selects the exact 14M
subset of [Figure 08's audited export](../data/14m-70m-quality-sparsity-latency.json).
It does not recompute losses, sparsity, ceilings, or timing statistics.
The one-row, two-panel [figure](../figures/22-14m-main-quality-sparsity-latency.pdf)
retains all 22 trained checkpoints and 20 Base/ReLU clipping settings, with
identical coordinates, colors, line styles, axis ranges and ceiling values.
The smaller canvas removes the 70M row and gives a 14M-only title and panel
labels. The shared legend retains Base, ReLU and the four T4/T7, Ph/Pall recipes.
[The new export](../data/14m-main-quality-sparsity-latency.json) records all
selected data, source/script/output hashes and the revised layout.

The original [Observation 025](025-matched-quality-latency.md) defines the
identity join and coverage: one seed, full validation over 338 complete blocks
from 500 documents, and retained K050 timings on RTX5090, BF16, batch one,
2,048 tokens and the full logit head. Trained timings use Run029/Run033;
clipping timings use Run036 and include recurring clipping costs. Eight
high-loss clipping points extend above the focused quality panel; all twenty
clipping latencies are visible. The caption discloses the quality tails and
points to the retained 14M clipping figure. Cross-session limitations remain.

## Caption and manuscript change

The caption describes (a) validation loss and (b) full-model latency against
model-wide sparsity for the same 22 Pythia-14M checkpoints. It identifies the
controls, pressure styles, threshold grid, clipping paths, ceilings and timing
workload. The introduction now explicitly focuses on Pythia-14M and calls
Pythia-70M a scale check. Its third observation reads:

> Third, sparsity remains exploitable across the tested scales: skipping
> computations can reduce latency at both 14M and 70M, but a sparse kernel
> does not transfer directly and requires adaptation to the larger model.

This is a claim about useful skipping, not a model-wide speedup that increases
with scale. The 14M evidence is the direct h/z ablation in
[Observation 034](034-operation-latency-manuscript.md). The 70M evidence is
[Run040 Observation 001](../../../runs/040-2026-09-20-pythia70m-kernel-overhead/observations/001-overhead-and-optimization.md):
at the retained T7/Ph checkpoint, disabling h/z skipping increases latency by
1.730113 ms in the paired diagnostic. The same study identifies the inefficient
dense h/z path and tests revised operation placement and sparse scheduling.
This one-checkpoint 70M diagnostic supports exploitability and adaptation;
it does not establish a scale law or a cohort-wide global speedup. No new
Run040 speedup numbers are added to the manuscript in this pass.

## Verification and limits

The four existing matched-quality/latency tests pass, covering source hashes,
checkpoint identity, integer counts, measured timing means and clipping tails.
Additional direct checks verify exact equality of all selected records to the
old 14M subset, no rendered 70M text, matching manuscript figure/data copies,
and unchanged historical figure hashes. The standalone PDF is visually checked
for readable labels, complete legend and lack of overlap. All introduction
cross-reference labels exist in the retained source.

The manuscript PDF hash and every other TeX source are unchanged. No LaTeX
build, model evaluation, benchmark, or cloud operation runs. The old two-size
figure/data remain as historical artifacts. Other manuscript sections retain
their current wording pending the author's later instructions.
