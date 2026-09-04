# Observation 001 — A0 14M raw-ELL compatibility

## Question

Can the pinned `sparser-faster-llms` implementation reproduce its supported
SparseLM inference result and then execute the exact Pythia-14M A0 `h -> W2`
shape faithfully enough to justify latency measurement?

## Method and coverage

Run 022 used upstream commit `661f1fc841ed84d92f3fca5e5a94cffc5bf00ee5`,
SparseLM0.5B revision `7c2a0473ec982facd5b21af81fe39c1783f3407e`,
an H100 NVL (compute capability 9.0; build target `sm_90a`; 95,830 MiB), CUDA
12.8, and the two pinned Python
environments in `config.yaml`. The upstream control retained batch 64,
sequence length 2,048, BF16, five warmups, and 50 repetitions.

The Pythia gate used signed BF16 ELL values with every nonzero retained, an
empty row, a fully dense row, integer row counts, and no allowed overflow. Its
declared correctness limit was relative L2 at most `0.02`. After the exact
`N=128` gate failed, a post-mortem repeated the same deterministic synthetic
matrix at four output widths and tested a padded adapter. This post-mortem is a
compatibility diagnostic, not an A0 benchmark.

Coverage stopped before loading the Pythia checkpoint for validation. Thus no
new A0 validation loss, activation occupancy, `R_model` relationship, or A0
runtime was measured. There is no figure or figure caption for this
observation.

## Results

| Test | TwELL/raw ms | Torch ms | Speedup | Relative L2 | Gate |
|---|---:|---:|---:|---:|---|
| Official SparseLM0.5B inference | 476.9542 | 620.0687 | 1.3001x | upstream control | pass |
| Raw ELL, `M=256, K=512, N=128` | — | — | — | 0.690548 | fail |
| Raw ELL, `N=256` | — | — | — | 0.001656 | pass |
| Raw ELL, `N=512` | — | — | — | 0.001657 | pass |
| Raw ELL, `N=2048` | — | — | — | 0.001654 | pass |
| Pad `N=128` through `N=256`, then slice | — | — | — | 0.001651 | diagnostic pass |

The width behavior is consistent with the pinned
`ell_spmm_warp_per_row` control flow. For `N=128`, only 16 lanes satisfy its
output-loop condition, while the inner reduction performs full-mask 32-lane
shuffles. Widths of at least 256 keep all 32 lanes participating. This is an
inference from the source and the controlled width sweep; it is not an
upstream-authored compatibility statement.

## Interpretation

The released high-level TwELL path is reproducible on its supported SparseLM
workload on H100 NVL. The released generic raw operator is not an unchanged
drop-in for Pythia-14M's 128-wide down projection, so Run 022 cannot supply the
desired A0 negative-control timing and cannot relate A0 `R_model` to speedup.

Pythia-70M's 512-wide down-projection width passed this synthetic gate, making
70M technically promising, but this does not validate its actual `K=2048`
checkpoint path, packing distribution, numerical behavior, or end-to-end
speed. The 128-to-256 padded adapter is a useful bound, not a fair exact-shape
runtime result, because it doubles the computed output width.

## Caveats

- The official control times are from one H100 NVL engineering Pod, not the
  final randomized fixed-SKU sweep.
- The width sweep tests operator correctness, not performance or actual 70M
  activation distributions.
- Two Python-side cast-before-mask repairs were needed because CUDA PyTorch
  cannot boolean-index `UInt16` directly. They preserve the guard and unpack
  mathematics and do not modify the upstream CUDA kernel.
- The first preflight hashes were computed from a Windows CRLF checkout. All
  eight expected hashes were proven to equal the clean Linux bytes after only
  LF-to-CRLF conversion; the Git commit and source text were unchanged.
- Attention was not tested. The upstream high-level converter implements MLP
  kernels, not QK/PV attention.

## Provenance

- Source orchestrator: `../05_start_worker.sh`
- Correctness gate: `../02_remote_preflight.py`
- Width diagnostic source:
  `../launch-control/h100nvl-20260904-110624/postmortem-raw-ell-widths.py`
- Official control CSV:
  `../artifacts/failed-gate-h100nvl-20260904/run022-output/attempts/003-20260904-143316/upstream-positive-control.csv`
- Width diagnostic JSON:
  `../artifacts/failed-gate-h100nvl-20260904/run022-state/postmortem-raw-ell-widths.json`
- Failed correctness log:
  `../artifacts/failed-gate-h100nvl-20260904/run022-state/remote-preflight-attempt004.log`
- CRLF/LF hash audit:
  `../launch-control/h100nvl-20260904-110624/attempt-001-crlf-hash-audit.json`
- Local-only hash-verified transfer archive (intentionally excluded from Git):
  `../launch-control/h100nvl-20260904-110624/run022-failed-gate-evidence.tar`
