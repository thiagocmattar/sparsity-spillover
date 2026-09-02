# Run 019 calibration launch record

This is a non-evidence infrastructure calibration. It does not authorize or
contain a scientific Run 019 condition.

## Launch identity

- implementation commit: `7a0c49f54a5d0b89943228490908417952766905`;
- complete Git bundle: 18,384,533 bytes, SHA-256
  `9bf4352bdfadcd17bc266176c7467ec09c816c22ecc509c7dbc60f1a178f75c3`;
- run-code SHA-256:
  `a46abdbfd95ae6267962238990d1a0301fbc17a6f5e535ef6b79df3b4072db86`;
- input-manifest SHA-256:
  `5127c0ebaafc7c7dff194a1ea5d628612221ad7958bb59b7f1c28d107488fe6d`;
- seven input files: 7,591,009,171 bytes;
- pinned container image:
  `runpod/pytorch@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35`.

The playbook originally called for a `git archive`, but remote setup requires a
real `.git` checkout. The launch therefore used one complete Git bundle whose
`main` ref resolves to the implementation commit. Both workers independently
verified the bundle hash and checked out that exact commit. The bundle's
symbolic `HEAD` was unset, so the workers explicitly checked out its verified
`main` ref; this changed no source bytes.

## Candidates and timeline

The requested Community RTX A6000 could not be provisioned in EU-SE-1 because
capacity was unavailable. The approved Secure A40 fallback and Secure A100 SXM
80 GB comparator were used.

| Candidate | Pod | Price | Transfer | Setup and strict init | Calibration | Result |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| A100 SXM 80 GB Secure | `uo3he5rjpjgmyx` | $1.59/h | 21:54:50--22:07:29Z (12m39s) | 22:09:09--22:14:48Z (5m39s) | 22:17:01--22:41:48Z (24m47s) | passed |
| A40 Secure | `4gf32uxfvdib2t` | $0.44/h | 21:54:52--22:58:42Z (1h03m50s) | 22:59:09--23:09:48Z (10m39s) | 23:12:17--00:08:11Z (55m54s) | passed |

Each setup rechecked all seven input hashes and independently strict-loaded the
canonical initialization twice. Every condition sample began with parameter
SHA-256
`76217bf2ef13de2377c7750515a559cb71f574c274476e08408a5f7749ea9cff`.
Neither calibration recorded a skipped optimizer update or non-finite loss.

The data payload was staged at `/workspace/data`. Because the checkout already
contains a tracked `data/` directory, the initial broad symlink nested beneath
that directory. Before setup, `data/tokenized` was linked explicitly and all
configured paths were re-hashed. On A100, the first split hash record is stored
as `data-sha256.txt` plus `input-sha256.txt`; the setup log then verifies all
seven configured paths together. A40 stores all seven in `input-sha256.txt`.

The A40 local upload was much slower than the A100 upload. A proposed temporary
cloud-to-cloud private-key transfer was rejected before any private key left the
local machine; the temporary local key files were removed. The original A40
upload completed normally.

## Retrieval and teardown

- A100 archive SHA-256:
  `01e64cade57ff1fc0cdebbdc4b11c001330239033da17f763e7554e5195708f1`;
  25 internal files verified.
- A40 archive SHA-256:
  `56641e9116c81f76788bcd606b7b22102ae67d085db31282c4aea3b006c63a92`;
  24 internal files verified.
- both calibration JSON files report `status: passed`;
- both initialization JSON files report `status: verified`;
- both Pods returned `deleted=true`, then `404_not_found`;
- both independent deadline guards were stopped after deletion;
- final account audit: zero Pods and zero serverless endpoints/workers;
- the pre-existing 100 GB network volume `9luykg5yc3` was neither attached nor
  modified.

See `resource-audit.json` for the final resource state and
`prelaunch/calibration/RESULTS.md` for the full-run cost/ETC comparison.
