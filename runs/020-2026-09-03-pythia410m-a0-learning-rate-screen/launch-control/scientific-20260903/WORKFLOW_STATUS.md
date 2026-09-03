# Run 020 scientific execution workflow

Launch approval was received on 2026-09-03 for two parallel full-pass A0 arms
under the documented maximum GPU-compute envelope of USD 73.44 plus minor Pod
storage. The immutable scientific source is commit
`90cf95d121efc782cb6cf86e50c12657dd279961`.

## Immutable transfers

- `tmp/run020-science.bundle`: 23,458,657 bytes, SHA-256
  `c0bc9336dcbcf91ce21842c843313a6e869a47ac7f3383e318defba587158ea1`.
- `tmp/run019-inputs.tar`: 7,591,015,936 bytes, SHA-256
  `eb6aba878d5e07c1ffad0df37ccd6f362e99e45e075b5640d383122e8ad900d5`.

## Workflow

1. DONE — design and launch approvals recorded; local tests passed.
2. DONE — live gate: zero GPU Pods/endpoints, USD 83.09 balance, hashes
   verified, and approved GPU capacity available.
3. DONE — provisioned two RTX PRO 6000 workers (one Community, one Secure) and
   armed independent exact-ID 10-hour deadline guards. Realized GPU rate is
   USD 3.78/hour; guarded maximum is USD 37.80 plus Pod storage.
4. DONE — established SSH and transferred/hash-checked source plus public
   inputs on both workers.
5. DONE — environment, GPU, initialization, schedule, and exact-dimension CUDA
   smoke gates passed on both workers.
6. FAILED CLOSED — both real worker entrypoints read the unresolved null
   top-level LR and exited before optimizer boundary 1. No training evidence was
   produced.
7. NOT STARTED — the failure occurred before the 30-minute training monitor.
8. NOT STARTED — no eligible terminal attempts exist for LR selection.
9. NOT STARTED — no selected checkpoint exists for TEAL.
10. DONE — retrieved and SHA-verified both failure/control archives, deleted
    both Pods, stopped both obsolete local guards, and confirmed zero GPU Pods
    and endpoints. Post-cleanup balance was USD 81.37 with USD 0.01/hour from
    the pre-existing retained network volume only.

## Failure archives

- `run020-failed-wh93a96ijgpeze.tar.gz`: 22,075 bytes, SHA-256
  `1d86b2c9a295d9429dc18ea364e3ebf2e00ad85fedfb5c90cdb5eadc99000d4b`.
- `run020-failed-l6vv7dxzhlfkuv.tar.gz`: 22,098 bytes, SHA-256
  `6793afafc8a61d38410be8b889d222178c213f062e8fc1df7d24fc1f8c7a86d1`.
