# H100 SXM attempt 001: frozen transfer

Status: **prepared, not yet launched**.

This is a fixed-policy hardware-transfer confirmation, not a new search. It
uses the six declared development endpoints for each of Pythia-14M, 70M, and
410M: A0, A1-H, A4-OL1 at `kappa` 0 and 0.5, and A7-OL1 at `kappa` 0 and 0.5.
The checkpoint, weights, gate, topology, sequence length, batch size, precision,
and validation contract remain unchanged. Only the deployment GPU changes.

## Measurement matrix

- P0, the minimally adapted Sakana starting point: 18 fresh processes.
- Frozen winner: K013 for 14M, K016 for 70M, and K010 for 410M, with three
  fresh processes per endpoint (54 processes).
- Frozen FFN-only and attention-projection-only contribution probes: 20
  processes. K010 is already z-only, so its projection component is its full
  winner result rather than a duplicate probe.
- Total: 92 evaluator processes.
- Each process uses 16 fixed timing inputs, five passes (80 paired samples),
  batch 1, sequence length 2,048, BF16, complete logits, and all 338 complete
  validation blocks. The excluded validation tail remains 1,444 tokens.
- P0 and frozen-winner repeat 1 also measure matched CUDA-graph native and
  candidate execution. Repeats 2 and 3 retain the primary eager estimand.
- QK-score and probability-value matmuls remain dense SDPA; "attention" in the
  component labels means only attention projection linears at sites a/z.

The winners are frozen before H100 observation. Failed numerical gates remain
failed observations; the controller records them and continues without
silently changing sites or policies.

## Source and transfer identity

The valid code bundle is `autoresearch/bundles/h100-code-004`, derived only from
Git commit `7a17285ca8782e53114cd9134a9daaedb45b8ded`. Its code archive is 499,482
bytes with SHA-256
`7f9823af9bf87e143cf8ca4ae5f1e9852e92f44ee8ee48c07444ff42ac96f377`.
The three pinned input archives plus code total 11,733,466,075 bytes. The remote
bootstrap verifies all four hashes before extraction and overlays committed
code after the older 14M calibration bundle.

Local bundle attempts 001--003 were never transferred and incurred no cloud
cost. Attempt 001 exposed an ordered-dictionary aggregation incompatibility;
attempts 002/003 exposed a Windows/.NET checksum-line writer incompatibility.
Bundle 004 passes an explicit LF-only checksum-byte check.

## Verification before launch

- Controller-specific tests: 8/8 pass.
- Relevant frozen-policy/evaluator tests, isolated to avoid the repository's
  duplicate candidate-test module names: 40/40 pass.
- Full bootstrap suite: 242/242 pass.
- PowerShell AST parsing passes for all three controllers.
- The valid archive contains the Phase 16 controller and bootstrap scripts at
  the expected paths.

## Compute envelope

The 2026-09-06 prelaunch catalog reports community H100 SXM availability HIGH at
$2.69/GPU-hour with CUDA 12.8 available. The launch uses one H100 SXM, a 60 GB
ephemeral volume, a 40 GB container disk, and a three-hour independent stop
guard. Maximum GPU cost is $8.07; no new persistent volume or endpoint is
created.

The 9.73 GB transfer previously took 280.3 seconds through the same encrypted
relay. Scaling that measurement gives about 5.6 minutes for all 11.73 GB,
before small setup overhead, or roughly $0.25 of H100 time. Expected total
wall-clock is 75--120 minutes including transfer, extraction, pinned-runtime
installation, evaluation, packaging, and retrieval ($3.36--$5.38 GPU cost).
The three-hour guard is a failure ceiling, not the ETC.

Launch follows completion, retrieval, and deletion of RTX PRO 4500 attempt 003,
so routine execution does not pay for two GPUs concurrently. After Phase 16,
the evidence archive and its SHA-256 are retrieved and verified before the H100
Pod is deleted. A final control-plane check must show zero Pods and endpoints;
the pre-existing 100 GB EUR-IS-1 standard volume remains the only intentional
billable resource.
