# Run 021 terminal results

Run 021 is complete and valid under the approved one-seed, one-pass A0
learning-rate screen. The predeclared selector retained the Run 019 peak
learning rate of `3e-4`.

## Selection and endpoints

Selection uses only the mean task loss over optimizer boundaries 649--712.
Paired validation loss and `R_model` come from the same complete eager
logical-product pass and do not enter the selection.

| Arm | Peak LR | Mean train loss, steps 649--712 | Final train loss | Paired validation loss | `R_model` | Clipped boundaries | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Run 019 A0 baseline | `3e-4` | **4.493717** | 4.448348 | 4.547456 | 0.000110 | 56 | selected |
| Run 021 A0 | `6e-4` | 4.513309 | 4.467776 | 4.552068 | 0.004222 | 13 | not selected |
| Run 021 A0 | `1e-3` | 6.851160 | 6.875180 | 6.786663 | 0.050171 | 20 | not selected |

All three arms were stable and eligible. The `6e-4` selection metric was
0.019592 (0.44%) worse than the baseline and its paired validation loss was
0.004612 (0.10%) worse. The `1e-3` arm was substantially worse on both
quantities. This refutes the narrow hypothesis that increasing peak LR from
`3e-4` to `6e-4` or `1e-3` fixes the 410M A0 endpoint under the matched
one-pass schedule.

The result does not establish that 410M is fully trained, nor does it rule out
the token-budget explanation. It tests no longer schedule, alternative warmup
or decay shape, lower LR, second data pass, or additional seed. The defensible
conclusion is that simple upward peak-LR retuning is not the remedy within the
approved grid.

## Matched identity and coverage

Both new arms completed 712 optimizer boundaries and 1,493,172,224 scheduled
input tokens from the same random-pretraining initialization, parameter
SHA-256 `76217bf2ef13de2377c7750515a559cb71f574c274476e08408a5f7749ea9cff`,
and schedule SHA-256
`d17a6c0c0d4aacff4b477e6d576f511c12c04ebbc37468f08e6fe61ff1c6ad8e`.
Neither arm overflowed or skipped an update. Every endpoint pass covered all
338 complete 2,048-token validation blocks (692,224 input tokens) and reported
the excluded 1,444-token tail.

| New arm | Attempt | Final SDPA validation loss | Median train tok/s | Train-loop time | Peak reserved GPU memory |
| --- | --- | ---: | ---: | ---: | ---: |
| `6e-4` | `001-20260903-160505-2b8d3334` | 4.552046 | 69,749 | 5.96 h | 21.13 GiB |
| `1e-3` | `002-20260903-160510-a3584498` | 6.786641 | 73,169 | 5.68 h | 21.13 GiB |

The small eager-versus-SDPA loss differences are execution-path diagnostics;
the canonical paper endpoint is the paired eager loss in the first table.
Mixed Community/Secure placement and throughput are execution provenance, not
scientific factors.

## TEAL disposition

No new TEAL computation was performed. The selected Run 019 baseline already
has status `complete_verified` with all ten target sparsities from 0.0 through
0.9 at
`runs/019-2026-09-01-pythia410m-selected-ladder-canonical-init/artifacts/teal/a0-gelu.json`
(SHA-256
`86028954797ed646962c2032a62a6bf6d79cfbdbae48f001a8a06e681b32ff80`).
That frontier spans paired loss 4.547456--9.027046 and `R_model`
0.000110--0.700471.

## Verification and closeout

- remote terminal verifier: `verified` for both conditions;
- local terminal verifier: `verified 2`;
- post-retrieval focused suite: 15 passed;
- post-retrieval full bootstrap suite: 183 passed;
- archive SHA-256:
  `2dd481ca8b3c73701cbb312013978e3bca5aceb9a0da79ad0314b35afc623733`
  (`6e-4`) and
  `ccec081d0ee5b0fb1c1b64eb9330312b08e5da403ed213364f31ca583e38975d`
  (`1e-3`);
- local file-level verification: 38/38 hashes matched for each attempt;
- locally retained attempt payload: 9,730,403,936 bytes, including
  9,728,622,900 checkpoint bytes;
- terminal RunPod inventory: zero GPU Pods and zero endpoints;
- post-teardown balance: $55.9091; ongoing spend is only the pre-existing
  unattached network volume at $0.01/hour.

RunPod's hourly billing endpoint had not yet emitted the final partial-hour
bucket at initial closeout. The exact terminal Pod cost is recorded only after
that bucket appears; the account-balance change and Pod lifetimes imply roughly
$25.3 total, safely below the $37.80 two-Pod GPU guard.

The other eleven 410M intervention conditions remain out of scope until this
A0 result and the final recipe are reviewed.
