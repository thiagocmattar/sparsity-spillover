# Attempt 005 launch adjustment

Attempt 005 starts from the original sealed Run-023 source plus the exact
Attempt-003 inference-context patch and this directory's eager-logical-context
patch. The latter matches the q/v lower-bound pass to Run 018's eager
logical-product protocol, restores SDPA Flash immediately afterward, and adds
explicit protocol labels to the output. It does not change checkpoints,
validation data or coverage, kernel arithmetic, timing contexts, repetitions,
or tolerances.

The full six-sentinel phase is rerun. No partial result from Attempts 001--003
is spliced into the final archive.
