# Preflight attempt 001

## Scope

This is the approved non-evidence Run 009 preflight. Scientific training is not
authorized in this attempt.

## Provisioning record

- Pod: `ujafdg00hqcq3q`
- Created: `2026-08-30T10:42:40.194Z`
- Cloud: Secure
- GPU: NVIDIA A100 80GB PCIe, verified 81,920 MiB
- CUDA host version: 12.8
- Data center: `CA-MTL-3`
- Live GPU price: $1.39/hour
- Persistent workspace: 25 GB at `/workspace`
- Manual teardown deadline: `2026-08-30T12:42:40.194Z`

The Pod was selected because it is the hardware-matched target for the lambda-1
condition and was available below the approved $1.59/hour ceiling. The source
packet SHA-256 is
`07c67aabd1a2bb562f143fe0622edced10229856445f2a19e2fab0d18c3676b1`.

## Current status

Passed and closed. The evidence packet was hash-verified locally, and the Pod
was stopped with `/workspace` retained for the lambda-1 worker. Scientific
execution remains unapproved.

## Result

- Five complete MB32/GAS32 lambda-1 boundaries passed.
- Boundary seconds: `7.3529`, `5.7323`, `5.7394`, `5.7424`, `5.7451`.
- Median boundary time: `5.7424` seconds.
- Peak reserved memory: `56.953125` GiB against a `71.326538` GiB limit.
- Final task/pressure loss: `10.678483` / `0.124259`.
- Final OL1 pressure-to-task ratio: `0.441796` with projection applied.
- FP16 overflow or skipped boundaries: zero.
- Initial parameters, schedule, config, code, validation cache, and training
  cache all matched the approved identities.
- Retrieval packet SHA-256:
  `3a89a18ec0a0451c6074fd852d1a8b305198cc4f68a627d6477bf67ffbe6a1c9`.
- Estimated compute charge: approximately `$0.90`; the RunPod billing bucket
  had not posted at closeout.

The retained Pod is `EXITED`, so GPU billing is inactive. Its 25 GB Pod volume
continues at stopped-storage rates until scientific reuse or the retention
guard terminates it at `2026-08-31T11:22:10.8696061Z`.
