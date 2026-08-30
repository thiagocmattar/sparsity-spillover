# Run 011 preflight attempt 001

## Result

Passed, retrieved, locally hash-verified, and stopped. No scientific worker ran.

- Completed: `2026-08-30T17:01:21.325415+00:00`
- Pod stopped: `2026-08-30T17:04:43.6536865Z`
- Control-plane runtime: 1,770 seconds
- Estimated GPU compute: `$0.78175` before posted-billing reconciliation
- Automatic deletion remains scheduled for `2026-08-30T18:04:31Z`

## Scope and authorization

The user explicitly approved one non-evidence Secure A100 SXM 80 GB preflight
on 2026-08-30. This attempt may run only `05_remote_preflight.py`: five exact
MB32/GAS32 boundaries at each of `kappa=0` and `kappa=0.5`. It may not start any
scientific `02_train.py` worker.

## Provisioned resource

- Pod ID: `n4e5u5ln2wyih9`
- Name: `run011-a4z-preflight-a100sxm-r1`
- Created: `2026-08-30T16:34:41.158Z`
- Secure `NVIDIA A100-SXM4-80GB`, one GPU, CUDA 12.8
- Live rate: `$1.59/GPU-hour`
- Image: pinned digest from `config.yaml`
- Container disk: 30 GB
- Persistent Pod volume: 25 GB at `/workspace`
- Existing network volume attached: no
- Automatic termination deadline: `2026-08-30T18:04:31Z`
- Maximum approved incremental envelope: `$2.60`

The control plane returned the Pod ID while its runtime was still null. The
resource is billable and must be tracked even if SSH readiness or setup fails.

## Source packet

Git bundle `source-5cb8afc.bundle` contains complete history ending at commit
`5cb8afc9eddd62a532f8a07c597f25e4f8128aa0`.

- Bytes: `7,652,566`
- SHA-256:
  `c6248e62a9a0da217602f7dafb926cef443a70885fd28495fa8e5cf20af47846`

The remote clone, config identity, and run-code identity must match before GPU
preflight work begins.

## Cache preparation

The first cache preparation tried a fresh build from the pinned public revisions.
At 750,080/1,000,000 training documents it reached the declared disk warning:
the 8.6 GB Hugging Face cache plus the growing token file left 1.3 GB free. The
non-evidence builder was stopped. Its exact 4,870,335,000-byte partial file and
the re-downloadable Hugging Face cache were removed; no published token cache or
scientific artifact was deleted.

The already-approved local cache was then transferred with SSH compression. Its
remote byte counts and SHA-256 hashes matched before GPU work:

- Train: 5,966,845,664 bytes,
  `da82a2ea2e0080c7fd681c7a93b07d3d9ff3d5357a8640895a82d536a1eaf97c`
- Validation: 2,774,672 bytes,
  `51cd758fda72f14383da30c358a895d0223c0d1d80b31455d2d842c3656d0451`

This cache copy is private to the stopped Pod. The separate pre-existing EUR
network volume already contains the same hash-verified master cache for future
compatible runs; this attempt neither attached nor modified that volume.

## Target-GPU checks

Both `kappa=0` and `kappa=0.5` completed five exact MB32/GAS32 optimizer
boundaries. Initialization and schedule hashes matched Run 004. Flash SDPA was
available. No gradient overflow or skipped boundary occurred.

| Metric | `kappa=0` | `kappa=0.5` |
| --- | ---: | ---: |
| Final smoke task loss | 10.641378 | 10.577690 |
| First boundary seconds | 5.402657 | 5.923452 |
| Median steady boundary seconds | 4.872744 | 4.870795 |
| Peak reserved GiB | 57.230 | 57.242 |

Across both endpoints, median steady throughput was about 430,490 input
tokens/second. Maximum reserved memory left 27.77% device headroom, above the
required 10%.

## Retrieval

`remote-preflight.json`, the preflight/setup/cache logs, and the pinned
environment freeze were copied locally. `transfer_inventory.json` records their
byte counts and hashes. Each local hash matched the corresponding remote file.
