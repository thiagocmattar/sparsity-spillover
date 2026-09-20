# Compact experimental setup and T/P baseline labels

## Question and scope

The user requested a compact experimental-setup appendix using the main
text's T/P terminology, without repeated definitions, data-packing details,
precision history or supplementary-file references. Preserve the training
settings table and make the effective batch and shared hyperparameters clear.
This is an editorial revision, with no new experiments or result estimates.

The training subsection is moved from the start of `results-appendix.tex`
into `experimental-appendix.tex`, so the complete setup is in one file.
The subsequent complete-results section is unchanged. The generic reachable-
operation set is replaced by the existing explicit Pythia counts. All
section, equation, table and figure reference labels remain available.

## Method and evidence

Training settings were checked against configs, manifests and metrics for
all 74 retained conditions, and Analysis018's `tables/training-protocol.md`.
The table preserves parameter counts, learning-rate ranges, microbatches,
accumulation and tokens per parameter. It now explicitly gives the effective
batch and common sequence length. At 14M, 32 sequences per microbatch times
32 accumulation steps gives 1,024 sequences per optimizer update. At 70M/410M,
4 times 256 gives the same effective batch. All sizes use 712 updates, 1%
linear warmup, the same AdamW beta/epsilon/weight decay, clipping limit,
zero dropout and seed values; 410M has a lower learning-rate range.

Ceiling formulas retain the same per-sequence counts and architecture
configurations. With FFN width four times hidden width, reachable counts
per layer remain 0, 4Td^2, 12Td^2 and the whole block count for T0, T1, T4
and T7, respectively; pressure placement does not alter these ceilings.
The site table gives shapes for one sequence instead of repeating a batch
dimension. T/P recipe coverage and the distinction between the single-site
L1/OL1 comparison and the multisite OL1 sweeps are preserved.

## Figure and caption

PDF: [19-base-model-optimization.pdf](../figures/19-base-model-optimization.pdf).
Builder: [30_relabel_base_optimization.py](../30_relabel_base_optimization.py).
Provenance: [base-model-optimization-labels.json](../data/base-model-optimization-labels.json).

**Base-model (T0/P0) training across model sizes.** (a) Training loss and
(b) gradient norm before clipping (logarithmic scale). Faint lines show
individual updates; solid lines show centered nine-update means, with
shorter windows at the ends. Gradient norms are computed after accumulation
and are not normalized by parameter count.

The PDF is copied to `manuscript/draft/figures/19-base-model-optimization.pdf`
and used in the training-and-evaluation subsection. Its source is Analysis021
Figure06 and Observation O007. The new builder reuses the original plot and
checks its data against every retained source log: three base models at
14M/70M/410M, 712 updates each, or 2,136 records in total. Only the two panel
titles and PDF metadata change from A0 to T0/P0; all twelve raw/smoothed
curves, colors, axes, limits and smoothing remain identical. Original
Analysis021 artifacts are unchanged.

## Results and limits

No numerical result changes. The same fixed training budget corresponds to
different token-to-parameter ratios, and the 410M learning-rate range differs.
These trajectories provide training context, not a convergence test or a
causal explanation of the cross-size loss differences. There is one training
seed per condition. Shared optimizer settings do not mean identical model
parameters or identical learning rates across sizes.

## Verification

The builder verifies exact equality with the original evidence export and
checks every plotted coordinate before and after relabeling. The PDF is
visually inspected. The revised TeX is checked for balanced structure,
resolved reference names, preserved training-table values and removal of
legacy terminology within the requested section. The user subsequently
authorized recompilation. The 34-page manuscript builds with resolved
references/citations and no overfull boxes; the revised appendix, training
table and figure are visually checked on pages 15--16. The reviewed build
replaces `manuscript/draft/main.pdf`; unrelated author source edits remain
unstaged, although the compiled PDF includes their current content.

Reproduce the figure from the repository root:

```powershell
.venv/Scripts/python.exe -X utf8 analyses/024-2026-09-17-h-only-kernel-latency/30_relabel_base_optimization.py
```
