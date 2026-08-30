# O001 — Local learning-rate pipeline calibration

## Question

Can the bootstrap execute and preserve a matched four-condition randomly
initialized Pythia-14M pretraining cohort on the local GPU under the approved
30-minute envelope?

## Method and coverage

- Peak LRs: `5e-4`, `1e-3`, `2e-3`, and `4e-3`; model and data-order seeds `0`.
- Pythia-14M topology `A0`, stock GELU, no activation pressure.
- 449 updates × 64 sequences × 2,048 tokens = 58,851,328 training input tokens
  per condition; identical schedule SHA-256 `d61d355668223d092d2d0f1b04daf9c614c45d6bffe2670ab4a6c63b1ae47523`.
- Every condition evaluated all 500 MiniPile validation documents after update
  1 and from the reloaded final checkpoint: 338 complete sequences, 692,224
  input tokens, and the declared 1,444-token excluded tail per pass.
- Final diagnostics covered exact/near-zero counts and moments for `a`, `m`, and
  `h` in all six layers, plus all 76 named parameter tensors.
- `03_verify.py` checked sequential events, finite metrics, matched identities,
  complete validation, diagnostic coverage, and every checkpoint file against
  its byte count and SHA-256 inventory.

## Result

| Peak LR | Final train loss | Final validation loss | Median step (s) | Median tokens/s |
| ---: | ---: | ---: | ---: | ---: |
| `5e-4` | 6.312839 | 6.257393 | 0.8497 | 154,254 |
| `1e-3` | 5.915539 | 5.839303 | 0.8427 | 155,535 |
| `2e-3` | 5.627964 | 5.542678 | 0.8444 | 155,226 |
| `4e-3` | 5.504993 | 5.418346 | 0.8371 | 156,574 |

All four attempts completed 449 updates, two full validation passes, final
diagnostics, checkpoint serialization/reload, and terminal artifact
publication. Total cohort wall time was 1,547.52 seconds (25m47.5s), within the
approved 30-minute envelope. The four checkpoint inventories cover 225,121,632
bytes in total.

Final validation loss decreased monotonically across this grid and was lowest
at its upper boundary, `4e-3`. This is a descriptive short-horizon ranking, not
an interior optimum or a learning-rate selection for later manuscript runs.

## Caveats and evidence label

Evidence is **provisional** only because the detached Python environment wrote
`null` for the Git commit and dirty-state fields in every attempt manifest. The
resolved configs, initial-parameter hash, cache hashes, schedule hash, and exact
six-file run-code inventory hash (`29160e3c…`) are present and identical across
all four attempts, so the executed inputs and code remain byte-identifiable.

This run has one seed, half the reference global batch, and a short horizon. It
does not test sparsity pressure, spillover, logical-product opportunity,
clipping, or runtime speedup.

## Source

- Training: `../02_train.py`
- Terminal verifier: `../03_verify.py`
- Machine-readable summary: `../artifacts/verification.json`
- Exact attempts: `../artifacts/attempts/`
