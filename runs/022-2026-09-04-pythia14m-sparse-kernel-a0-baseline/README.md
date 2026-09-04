# Run 022 — Pythia-14M A0 sparse-kernel baseline

**Status:** executed 2026-09-04; stopped at the raw-ELL correctness gate; evidence retrieved; RunPod GPU terminated.

## Question

Can the released Sparse-er/Faster-LLMs (`sparser-faster-llms`) implementation be reproduced on its supported SparseLM workload, and can its low-level TwELL ELL-SpMM primitive execute the exact Pythia-14M A0 `h -> W2` operation without changing the model output? This is the first compatibility and measurement step toward relating the repository's logical `R_model` statistic to measured runtime on the selected 14M and 70M endpoints.

Run 022 does **not** test a sparsity speedup hypothesis for A0. A0 uses GELU and is expected to have essentially no exact zeros; its raw-ELL path is a deliberately adverse negative control. It establishes correctness, overhead, occupancy, and measurement contracts before any sparse endpoint is benchmarked.

## Fixed design

- Source checkpoint: Run 004 `gelu-control`, seed 1234, step 712, after exactly one MiniPile pass (1,493,172,224 training tokens).
- Architecture: Pythia-14M, A0, six layers, `d_model=128`, `d_mlp=512`, sequence length 2,048.
- Validation: all 500 MiniPile validation documents; all 338 complete 2,048-token blocks (692,224 tokens); the 1,444-token tail remains explicitly excluded.
- Source-model fidelity check: FP32 parameters under FP16 autocast and Flash SDPA must reproduce the archived validation loss `5.208582500028893` within `2e-4`.
- Runtime representation: BF16, because the released CUDA operator accepts BF16.
- Hardware: Hopper compute capability 9.0 and CUDA 12.8. Final comparisons use one exact GPU SKU.
- Upstream positive control: unmodified `benchmark_inference.py` at commit `661f1fc...`, model `SakanaAI/SparseLM0.5B` at revision `7c2a047...`, with the upstream default `(batch=64, sequence=2048, BF16, 5 warmups, 50 repetitions)` workload.
- Pythia dense control: ordinary Transformers inference and an identity-hook version, batch sizes 1 and 32, using deterministic real validation blocks.
- Pythia raw-ELL negative control: every nonzero `h` value is packed exactly; no pruning, capacity truncation, or silent overflow is allowed. Each of six `h -> W2` operations is checked and timed at 256 and 2,048 rows.

Two isolated Python environments are required. The upstream control retains its pinned Torch 2.9.1 / Transformers 4.56.2 stack. The Pythia measurement retains this repository's Torch 2.11.0 / Transformers 5.12.1 stack. Sharing one environment would confound code compatibility with a dependency migration.

## Measurements

The run records:

- archived-loss reproduction and full BF16 validation loss;
- exact-zero and near-zero integer counts at `h`, pooled before division, for every layer;
- RMS, global/row L2, row-NNZ and 256-wide tile-NNZ distributions;
- TwELL payload-overflow incidence at factors 8, 4, and 2;
- dense and identity-instrumented full-model latency/throughput;
- exact ELL pack, sparse-kernel-only, and pack-plus-kernel latency by layer and row count;
- dense-BF16 and FP32-reference numerical errors; and
- GPU/runtime/environment identity and raw timing blocks.

The run-local `R_covered` diagnostic is only the canonical integer numerator attributable to operations actually covered by a tested kernel divided by the unchanged canonical `R_model` denominator. It does not replace `R_model` and is not a runtime-gain estimate. Run 022 does not report `R_covered` for A0 unless the canonical count inputs are present in the result artifact.

## Stop rule and interpretation

The stage succeeds when the unmodified upstream positive control runs and TwELL beats its Torch baseline, the Pythia checkpoint and validation identity pass, and the raw ELL operation is numerically faithful with zero skipped rows. The raw ELL path may be—and for dense A0 is expected to be—slower than dense GEMM.

Any upstream-control failure, checkpoint/loss mismatch, non-Hopper execution, source-hash mismatch, dropped ELL value, overflowed row, or relative-L2 error above 0.02 stops promotion. Results are reported before creating the future A1-H run.

Attention is outside Run 022. The released high-level TwELL replacement covers MLPs, and its high-level packing assumes ReLU-positive activations. The generic low-level ELL operator accepts signed BF16 values, so a later approved A7 study can test partial QK/PV compositions; it is not a drop-in attention kernel and cannot exploit Q and K zeros jointly. Logical QK/PV opportunity is not evidence that the released kernel accelerates attention.

No gradient-interaction diagnostic is collected because this is forward-only inference on an already trained checkpoint; such diagnostics cannot be reconstructed here and are not needed for the runtime question. No model or optimizer checkpoint is written. The hash-pinned Run-004 source checkpoint remains retained locally. Clipping is disabled in Run 022; the stored A0/A1-H TEAL operating points belong to the separately gated follow-up stage.

## Files and outputs

- `config.yaml`: immutable scientific and software identities.
- `benchmark_core.py`: exact ELL and count-first measurement primitives.
- `00_prepare_inputs.py`: allowlisted transfer bundle and hash inventory.
- `01_static_preflight.py`: local checkpoint/cache/source validation.
- `02_remote_preflight.py`: Hopper/CUDA/operator correctness gate.
- `03_benchmark.py`: Pythia validation, occupancy, and timing experiment.
- `04_verify.py`: fail-closed result verification.
- `05_start_worker.sh`: disconnect-safe remote orchestration.
- `06_monitor.py`: one read-only monitoring snapshot.
- `00_setup_remote.sh`: pinned two-environment setup.
- `DEPLOYMENT_PLAYBOOK.md`: proposed launch/retrieval/teardown sequence.

Pre-launch generated inventories live under `prelaunch/`. Remote results will live under `artifacts/attempts/<attempt-id>/`; no result directory is fabricated before launch.

## Execution outcome

The official SparseLM0.5B positive control passed on an H100 NVL: TwELL took
`476.9542 ms` versus `620.0687 ms` for Torch (`1.3001x`). The exact Pythia-14M
raw operation did not pass correctness. At `M=256, K=512, N=128`, the released
`ell_spmm_raw` operator had relative-L2 error `0.69055`, above the declared
`0.02` limit. The run stopped before Pythia validation, occupancy collection,
or A0 timing, as required by the stop rule.

A bounded post-mortem isolated the incompatibility to the small output width:
the unchanged operator was correct at `N=256`, `512`, and `2048` (relative-L2
approximately `0.00165`). Padding `N=128` to `256` and slicing was also correct,
but is not the exact 14M operation and doubles the output work. See
`observations/001-a0-14m-raw-ell-compatibility.md` and
`POSTMORTEM_AND_REVISED_PLAN.md` for evidence, interpretation, costs, and the
proposed next gate.

RunPod Pod `n3ovkit8h3y2wh` ran for approximately 34 minutes at `$2.59/hour`.
The observed account debit was `$1.47576`, including transient storage and the
pre-existing storage draw. The evidence archive was hash-verified locally
before termination, and RunPod reported zero active Pods after teardown. The
immutable `config.yaml` retains its pre-launch status string as provenance;
this README records the actual terminal state.
