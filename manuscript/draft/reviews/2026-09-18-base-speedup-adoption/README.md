# Base-model speedup figure and agentic kernel introduction

The author requested Analysis024 Figure13 immediately after the post-hoc
calibration paragraph in `experimental-study.tex`, together with an accessible
account of agentic kernel development, the sparsity/speedup relationship and
the limitations of clipping under the final kernel.

The [unchanged figure](../../figures/13-14m-70m-sparsity-base-speedup.pdf)
is now Figure 3 on page 5 of [main.pdf](../../main.pdf). It appears directly
after the requested paragraph. Four short paragraphs on pages 5-6 introduce
the development workflow and execution mechanisms before Section 4.1. Later
figure numbers and references are resolved automatically. The updated draft
has 33 pages. No analysis figure or benchmark measurement changed.

## Evidence and wording

- [Figure observation and builder](../../../../analyses/024-2026-09-17-h-only-kernel-latency/observations/030-base-speedup-clipping.md)
  define all 44 trained and 40 clipping points. The PDF and its exact data export
  were copied unchanged, with hashes in the figure/data source manifests.
- [Run029 implementation audit](../../../../runs/029-2026-09-07-pythia14m-matched-kernel-retrospective/observations/06-implementation-and-claim-audit.md)
  supports the human-guided development trajectory and GPT-6 Astra configured
  default. The main text names that configuration; the existing appendix
  retains the distinction from per-request served-model attestation. The
  author's statement that the team lacks specialist systems expertise is
  presented as author context, not an independently measured agent comparison.
- [Run036 protocol](../../../../runs/036-2026-09-18-controls-clipping-final-kernel/README.md)
  and [adapter](../../../../runs/036-2026-09-18-controls-clipping-final-kernel/clipping_adapter.py)
  establish separate, timed masking operations around frozen kernels. The
  [operation evidence](../../../../analyses/024-2026-09-17-h-only-kernel-latency/observations/027-operation-bypass.md)
  explains skippable groups, short rows, avoided weight reads and retained work.
  These code mechanisms do not isolate each overhead's timing contribution.

Two requested claims needed qualification to match the retained data:

1. **Linearity is within recipe families.** An OLS fit with an intercept for
   each five-threshold family gives positive slopes and R-squared between
   0.908 and 1.000. Pooling all 22 trained checkpoints at each size instead
   gives R-squared 0.480 at 14M and 0.445 at 70M. The text therefore describes
   an approximately linear within-family increase and different slopes,
   without a universal or proportional speed law. It gives the 14M kappa=.5
   T7/Pall versus T4/Pall counterexample: 27.48% versus 12.71% sparsity, but
   1.38x versus 1.42x speedup. These are fixed-grid descriptions, not independent
   seed replicates; no fit line, inferential test or confidence interval is added.
2. **Clipping gains are limited, not universally absent.** All 18 positive
   clipping settings at 14M remain below the plotted base reference. At 70M,
   five positive settings exceed it, with a maximum 1.221545x for the base
   model at p=.9, at retained loss 9.179397. The main text says aggressive
   70M clipping can reach 1.22x at substantial quality cost and links to the
   full-range clipping appendix. It does not misrepresent this as no gain,
   nor claim a fused clipping implementation was optimized.

The caption explicitly defines the denominator as the final specialized
implementation's base-model latency, not native execution. It preserves the
cross-session, quality and different-panel-scale limitations. The narrative
connects scalar zeros to saved arithmetic/data movement and emphasizes that
location, grouping and the implementation determine which work is avoided.

## Verification

All 84 ratios agree with the unrounded latencies and the two common references.
All 34 manuscript figure copies match their manifest hashes. The new figure
and data copy match the analysis byte-for-byte. [verification.json](verification.json)
records the numerical checks, descriptive fits, source hashes and resolved
figure/section labels.

The isolated build used pdflatex, BibTeX and two further pdflatex passes.
All references and citations resolve; there are no overfull boxes. Six
underfull vertical boxes and one underfull horizontal box remain. Pages 5-6
were inspected at high resolution and all 33 pages in rendered contact sheets;
no clipping or overlap was found. QA files and intermediate builds remain
ignored under `tmp/pdfs/base-speedup-adoption/`. No training, benchmarking,
cloud execution or new scientific measurement ran.
