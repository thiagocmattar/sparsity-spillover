# Fixed thresholds and specialized kernels: argument revision

User-authorized manuscript revision, 8 September 2026. No experiment,
kernel implementation, numerical table or figure artwork was changed.

## Argument map

1. Q-Sparse controls retained activations with magnitude top-k. Spark motivates
   statistical top-k by sorting-based selection overhead. A fixed threshold
   avoids ranking and input-dependent moment estimates, while leaving the
   retained count free to change during training.
2. Extend the FFN elementwise-nonlinearity design to symmetric hard thresholds
   on attention's Q/K/V coordinates. The learned distribution's adaptation is
   the tested hypothesis; a prescribed sparsity fraction is not imposed.
3. Converting this opportunity into execution requires compatible shapes,
   nonlinearities and hardware. The upstream Sakana control and adapted P0
   are distinct tests; neither demonstrates unchanged upstream acceleration
   for this Pythia-14M workload.
4. Human-guided coding-agent search makes specialization practical within the
   architecture study. Prior execution-guided kernel work motivates the method;
   this study supplies a qualified result, not a comparison of agent ability.
5. The matched retrospective reaches 1.6024x at iteration 11 and 1.7830x at
   iteration 42. The selected implementation's 30-checkpoint speedup has a
   strong approximately linear association with model-wide sparsity.
6. Ablations identify fusion and conditional sparse-path gains. Attention
   skipping loses time despite skipping many MMA instructions. Either entirely
   zero operand fragment permits a skip; matching zeros in both Q/K are not
   necessary and no profitable joint-mask threshold has been identified.

## Files and sources

- `methodology.tex`: motivation and fixed-threshold adaptation hypothesis.
  The existing mathematical definitions and derivative conventions remain.
- `kernel-autoresearch.tex`: the six-step argument, current 30-checkpoint
  summaries, early versus late progress, precise attention limitation. The
  caption now identifies the progress curve as a matched retrospective.
- `results-appendix.tex`: upstream compatibility versus adapted P0, actual
  four-checkpoint historical replay versus final 35/current 30 sweep, and
  instruction-level skip semantics.
- [Literature review](literature-argument.md): primary Q-Sparse and Spark
  sources. No full-sort requirement, speed comparison, or asymptotic advantage
  over Spark is claimed. The unchanged related-work section already distinguishes
  attention-position selection from Q/K/V feature sparsity.
- [Kernel evidence audit](kernel-evidence.md): exact observations and source
  paths in Runs 022, 025 and 029; current reduction from Analysis 018.
- [Final scientific review](final-scientific.md): the replay-coverage issue
  and two terminology suggestions were implemented and verified.
- [Reader review](final-reader.md): nine affected pages pass visual/readability
  review on the penultimate PDF. The final edit replaces "establish no" with
  "do not identify" in the appendix; only page 26 changes visually and was
  inspected again by the primary agent. All other page PNGs are identical.

The authors' expertise is framed through the study's purpose and the practical
role of agent assistance, rather than a personal disclaimer. Approximate
linearity is descriptive; early improvements are not evidence of convergence.
The unchanged-source Pythia primitive failed before full-model timing, so its
failure must not be described as an unchanged Sakana full-model speedup.

## Verification

`verification.json` records the final 27-page PDF, unchanged introduction and
bibliography, all figure/data/source hashes, exact table preservation, resolved
references and clean build. New pages 1-2 and 14-25 have pixel-identical bodies
to the previously inspected reading copy (ignoring page-number footers).
The primary agent inspected pages 6-9 and final page 26; the reader agent
inspected pages 3-5, 10-13 and 26-27. Figure typography and artwork are unchanged.
This is a reading wrapper; ICLR submission-template fitting is not assessed.

Final PDF SHA-256:
`75b35b3dfcc4dd49b0c826ad6e218bd0d265ed2e04a0df847818e6d74ff5de05`.
The new analysis-owned snapshot preserves the sources while the draft stays
local-only. Previous snapshots and run records remain unchanged.
