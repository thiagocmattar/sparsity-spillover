# Approved Run042 launch

User authorization: new iterative 70M optimization launch, four hours and USD15,
full-forward native dense-base reference and gain decomposition. Goal started
2026-09-20T13:47:59Z; absolute stop deadline2026-09-20T17:47:59Z. The first
six immutable candidates span M8/M16 and N128/N256/N512 h/z layouts. Search
may add evidence-motivated candidates within the declared32-candidate/time cap.

Seven focused CPU checks pass: thresholds/rounding, norm/RoPE gate restoration,
retained cohort/coverage, and scalar versus issued/bypassed instruction counts.
All242 bootstrap checks pass. The first bootstrap invocation used an overly
long Windows temporary path and produced five file-creation errors; the full
suite passes using the short isolated tmp/r42boot directory. No bounds changed.
All1435 retained input/source copies are hash-verified. GPU checks are pending:
28 original operators,24 training-block full-model comparisons, six smokes,
and48 synthetic output/counter cases per tile variant. A failed candidate is
recorded and cannot enter selection; a baseline/smoke failure halts that phase.

One RTX5090,32GB, Secure EU-RO-1 if still available, USD0.99/hour from the live
catalog. Pinned image is the verified Run040 digest:
runpod/pytorch@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35.
80GB persistent Pod storage at /workspace;20GB container; SSH only. Torch2.11.0
with CUDA12.8, Transformers5.12.1 and exact prior lock. Local GPU differs from
the paper benchmark class, so use the authorized cloud resource.

Expected setup/transfer10-20min, initial measurements15-30min, bounded search
until the final-evaluation reserve, then final controls/recovery20-30min.
Maximum compute USD3.96 for four hours, plus temporary disk, with a hard USD15
task cap. Stop earlier on a verified result. No network volume or endpoint is
created. Existing Run041 and the shared network volume are preserved.

Retain the complete README diagnostic inventory, both sizes' checkpoint/cache
identities, all trials, raw timings and block-level outputs checks. New timings
use one GPU UUID and common native references per size. Sources are committed
before launch; candidates remain immutable after registration. Detached logs
and artifacts live on /workspace. Monitor every60 seconds, with refreshed
phase progress, validation loss, throughput, ETC and spend; investigate failed
numerics, stale activity, low memory/disk. A scoped local Pod stop guard and
credential-free remote job deadlines enforce the time limit. Provider
credentials remain local. Delete only after archive and member hashes verify.

No further approval is needed within the user's explicit launch authorization.
No manuscript edit or PDF rebuild is included.
