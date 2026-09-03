# Run 021 scientific execution workflow

Launch approval was received on 2026-09-03 for the corrected Run 021 under the
documented maximum GPU-compute envelope of USD 73.44 plus minor Pod storage.
The immutable scientific source is commit
`809d7668d10de3300fa0d1555e71f60f5145e8d5`.

## Immutable transfers

- `tmp/run021-science.bundle`: 23,503,741 bytes, SHA-256
  `4c0dd59dcb826f7a3b4005344ef5e199d0460ac14098c46c729367a8b785ab96`.
- `tmp/run019-inputs.tar`: 7,591,015,936 bytes, SHA-256
  `eb6aba878d5e07c1ffad0df37ccd6f362e99e45e075b5640d383122e8ad900d5`.

## Workflow

1. DONE — local focused, bootstrap, initialization, entrypoint, and shell
   verification passed; source committed and bundled.
2. DONE — live gate found zero Pods/endpoints, sufficient balance, and approved
   capacity.
3. DONE — provisioned two RTX PRO 6000 workers (one Community, one Secure) and
   armed independent exact-ID 10-hour deletion guards. Realized GPU rate is
   USD 3.78/hour; guarded GPU maximum is USD 37.80 plus Pod storage.
4. DONE — source and input payloads transferred and SHA-256 verified on both
   workers; the temporary pod-to-pod relay key was removed.
5. DONE — setup, per-worker preflight, corrected LR dispatch, and full-shape
   CUDA smoke passed on both workers.
6. DONE — detached `a0-lr-6e-4` and `a0-lr-1e-3` workers completed all 712
   optimizer boundaries with distinct condition-resolved LRs, finite diagnostics,
   no overflow, and no skipped update.
7. DONE — monitored both workers through complete validation; both terminal
   attempts passed the remote and local scientific verifiers.
8. DONE — retrieved both attempts; both archive hashes and all 38 file-level
   hashes per attempt matched. The predeclared selector retained Run 019 A0 at
   peak LR `3e-4`.
9. DONE — because the Run 019 baseline won, no Run 021 TEAL job was started;
   the already-complete verified Run 019 A0 ten-point frontier is reused. Both
   Run 021 Pods were deleted.
10. DONE — all agreed artifacts are local, both guards were stopped, and the
    terminal control-plane check returned zero GPU Pods and zero endpoints.

## First real-boundary gate

At optimizer step 2, `a0-lr-6e-4` reported LR
`8.426966292134831e-05`, task loss `11.012524865567684`, and 70,218
tokens/second. `a0-lr-1e-3` reported LR `0.0001404494382022472`, the same task
loss, and 72,760 tokens/second. Both reported finite gradients, clipping to
1.0, no overflow, and no skipped update.

## Terminal result

The selection metric was the mean task loss over boundaries 649--712. It was
`4.4937172935` for the pinned Run 019 `3e-4` baseline, `4.5133089319` for
`6e-4`, and `6.8511603217` for `1e-3`. All arms were eligible and stable, so
the selector retained `3e-4`. Paired eager full-validation loss was 4.547456,
4.552068, and 6.786663 respectively. The result refutes a simple higher-LR
explanation within this grid; it does not test a longer token budget.
