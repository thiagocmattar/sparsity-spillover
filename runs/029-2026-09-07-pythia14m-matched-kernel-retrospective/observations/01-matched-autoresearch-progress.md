# Matched retrospective specialization progress

## Question and method

Can the retained development history demonstrate a qualified full-model gain
without mixing its original benchmark denominators? Replay the archived
implementations on one physical RTX5090, with a fixed c30 checkpoint
(A7+OL1, kappa0.5), BF16, batch1, sequence2048 and all50304 output logits.
The reference is native dense PyTorch/SDPA in the shared CUDA-graph scaffold,
not stock eager latency. Stock eager output independently anchors qualification.
Recurring input-dependent packing, zero checks, model execution and full logits
are timed; graph construction/compilation and equal input staging are excluded.

Each score equally pools1344 paired host-latency ratios:64 fixed validation
inputs,7 randomized paired passes,3 fresh processes. Qualification requires
all338 complete blocks from all500 MiniPile validation documents,691886
prediction tokens and the declared1444-token excluded tail. Bounds are
elementwise atol0.25/rtol0.02, relative-L2 at most0.02, finite outputs and pooled
loss difference at most0.001. No weights, gates, thresholds or kernels were tuned.

## Coverage and result

All50 K proposals are archived;42 are eligible14M full-model proposals.
K011's12 retained masks occupy the same proposal ordinal, yielding53 K
configurations. Eight other-size-only/standalone proposals are excluded with
reasons in `provenance/candidates.json`; they are not assigned invented timings.
P0 is a separately benchmarked reference, not a search iteration.

The progress score is the cumulative maximum of qualified c30 scores and the
native1x reference. Thus unsuccessful/slow candidates leave it unchanged.
There are four improvements:

| Paper iteration | Archived candidate | Qualified speedup |
|---:|---|---:|
| 9 | K017 | 1.057046x |
| 10 | K018 | 1.097161x |
| 11 | K019 | 1.602386x |
| 42 | K050 | 1.783029x |

Iterations1-8 do not improve on native; they have not been shifted to1x.
K050's three independent c30 progress replicates span1.782286-1.783693x.
Its separate final-matrix c30 evaluation gives1.783175x, without pooling those
additional processes into the predeclared progress score.

## Figure and caption

Current paper asset: [01-matched-autoresearch-progress-r02.pdf](../figures/01-matched-autoresearch-progress-r02.pdf).

**Caption:** Matched retrospective progress in agent-guided kernel specialization
for Pythia14M on RTX5090. The line is the best fully qualified full-model
speedup found through each eligible proposal, on one fixed sparse checkpoint.
The native implementation supplies the genuine1x starting reference before
proposal1 (plotted at0). Markers indicate incumbent improvements. All candidate
timings share the same native CUDA-graph denominator and numerical protocol.

## Limits and provenance

This is a retrospective replay of one development history, not a newly executed
search or independent trials. The cumulative maximum is monotonic by definition;
it does not imply every candidate improved. One fixed checkpoint defines the
selection metric, not an average over changing model sets. Generalization across
the35 checkpoints and sparse-path attribution are addressed in Observation03.
It is not evidence of superiority to other agents or human developers.

Source scripts: `03_matrix.py`, `10_reduce.py`, `12_figures.py --revision 2`,
`16_report.py`. Source data: `results/matched-retrospective-001.json` and
`results/report-001.json`; raw source hashes and the full plan are linked there.
`results/figures-002.json` records publication hashes. Revision1 PDF is retained
as a superseded proof; revision2 moves the native-reference annotation clear
of the early steps and corrects numeric tick formatting. Data are unchanged.
