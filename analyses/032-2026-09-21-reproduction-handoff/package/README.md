# Activation Sparsity as Training–Inference Co-Design

A supplementary code and evidence companion for **Activation Sparsity as
Training–Inference Co-Design**. It contains the methods, executed condition
grid, final CUDA implementations and compact measurements behind the paper.
It does not contain model weights, tokenized data, cloud tooling or the experiment
development history.

Read [the manuscript](main.pdf) and [the paper-to-code map](docs/PAPER_MAP.md). Humans and agents use the
same entry points; [AGENTS.md](AGENTS.md) records the scientific contracts.

## Quick start

Use Python 3.12. For CPU review, install a compatible PyTorch build first;
for GPU reproduction use [the recorded environment](environment/README.md).

```bash
python -m venv .venv
# Linux: source .venv/bin/activate
# PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python scripts/reproduce.py verify
python -m pytest -q
python scripts/reproduce.py results
python scripts/reproduce.py figures
python -m training.train --list
```

These commands need neither weights nor a GPU. `reproduced/` contains a CSV of
all **95 endpoints (45/11/27/12 at 14M/31M/70M/410M)**, the 36 execution
coordinates in the three-scale comparison, numerical checks, and central
figure reconstructions. The manuscript is in `main.pdf`; exact paper figure
assets are in `figures/`. Start with [the reviewer guide](docs/REVIEWER_GUIDE.md)
for the claim-to-evidence map and a short CPU-only review route.
Reconstructed plots use the same retained numbers; typography is not a
byte-for-byte reproduction of the manuscript artwork.

## Rerun experiments

[REPRODUCE.md](docs/REPRODUCE.md) provides data preparation, a CPU smoke,
one-condition training, full diagnostics, clipping and GPU qualification commands.
The condition list is the executed grid, not the full combinatorial search space.

The portable training and benchmark entry points were extracted/refactored for
this release. Core accounting, gates, OL1 math and CUDA arithmetic retain source
provenance. CPU tests do not establish a new GPU qualification; run the included
full-validation gate before reporting any new latency.

**Initialization matters.** Historical runs share exact random tensors. These
are omitted along with model weights. Supply the canonical untrained tensors
and pass their parameter-hash check for an initialization-matched rerun, or use
`--fresh-initialization` for an explicitly new seed-matched replication. The
latter must not be described as a byte-exact replay of the reported trajectory.
There is no public weight download URL in this snapshot.

## Contents

| Path | Purpose |
|---|---|
| `src/sparsity_research/` | Gates, hooks, pressure, exact counts, ceilings, validation |
| `training/` | Small training/evaluation entry points and retained optimizer boundary |
| `configs/` | Architecture pins and the resolved paper grid |
| `kernels/` | Final 14M/31M/70M components, direct assembly, published ablations |
| `scripts/` | Data preparation, CPU result reconstruction, clipping, GPU measurement |
| `results/` | Compact measured values, integer counts and diagnostic traces |
| `main.pdf` | Complete manuscript, including appendices |
| `figures/` | Six exact figure assets used by the manuscript |
| `docs/` | Paper map, mathematics, reproducibility limits and working guide |
| `tests/` | Mathematical behavior, data coverage, hook placement and portable interfaces |
| `PROVENANCE.json`, `MANIFEST.json` | Source hashes and release SHA-256 inventory |

Configs use scientific condition names such as `14M-T2-Ph-0.1`. Kernel folders
are organized by model size and operation. No original checkout is needed.

## License and citation

The authors have **not selected a project license**. No new license grant is
implied. Third-party notices are preserved in [LICENSES/](LICENSES/) and
[THIRD_PARTY.md](THIRD_PARTY.md). Cite the paper title above; author, venue and
persistent publication identifiers should be added when finalized.

See [release verification and limitations](docs/RELEASE_STATUS.md) before running
expensive experiments or interpreting the evidence.
