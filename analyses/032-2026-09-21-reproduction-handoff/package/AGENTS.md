# Agent guide

Read README.md, docs/PAPER_MAP.md, docs/REPRODUCE.md and docs/RELEASE_STATUS.md.
Use docs/METHODS.md for the executed definitions and the separately submitted paper
for its narrative. The manuscript is not bundled; its writing may evolve independently.

- This is a reproducibility companion. Do not invent a new scientific comparison.
- CPU checks and reconstruction from results/ are safe default tasks. Ask the
  human to approve a concrete design and launch before long training or paid GPU
  work; state hardware, estimated time, storage, diagnostics and cost envelope.
- Never load released Pythia weights for a random-pretraining experiment.
- A new random draw is a new replication, even with seed 1234. Preserve the
  initial parameter and data-order hashes and keep all compared conditions matched.
- Gates and pressure targets are independent. T2 is HZ=(h,z). T7 Q/K are post-RoPE. Gate equality survives.
- Keep all 500 validation documents: 338 complete 2,048-token blocks, 1,444-token
  excluded tail. Sum integer counts before dividing.
- Preserve ordinary loss, logical-pass loss, activation zeros, logical products,
  MMA bypass and latency as different quantities. Never infer speed from sparsity.
- Training uses dynamic FP16 with FP32 parameters; the kernel test uses BF16.
- Record gradient interaction during training. Checkpoints cannot recover it.
- All final timing candidates must pass the full numerical gate in three fresh
  processes. Retain failed checks and raw timings. Keep sessions separate.
- results/ is read-only evidence. Write new measurements to a new outputs/ folder
  and new reconstructions to reproduced/. Do not replace the paper's numbers.
- Run focused tests after edits. CPU success never certifies CUDA correctness.
- Keep this release lean: no historical run trees, providers, credentials, token
  caches, model weights or compiled extensions in version control.
