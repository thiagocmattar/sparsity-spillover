# Where Sparsity Matters

A lean code companion for **Where Sparsity Matters: Shaping Activations for
Efficient Transformer Execution**. It contains the methods, executed condition
grid, final CUDA implementations and compact measurements behind the paper.
It does not contain model weights, tokenized data, cloud tooling or the experiment
development history.

Start with [the paper-to-code map](docs/PAPER_MAP.md). Humans and agents use the
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
all **84 endpoints (45/27/12 at 14M/70M/410M)**, numerical checks, and central
figure reconstructions. Original paper PDFs remain in `results/paper-figures/`.
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
| `configs/` | Architecture pins, paper grid, original scientific settings |
| `kernels/` | Final K050/opt073 and their required components; pinned dependency fetcher |
| `scripts/` | Data preparation, CPU result reconstruction, clipping, GPU measurement |
| `results/` | Compact measured values, integer counts, diagnostic traces and reference PDFs |
| `docs/` | Paper map, mathematics, reproducibility limits and working guide |
| `tests/` | Mathematical behavior, data coverage, hook placement and portable interfaces |
| `PROVENANCE.json`, `MANIFEST.json` | Source mapping and release SHA-256 inventory |

The original repository's run names in provenance are historical identifiers,
not missing runtime dependencies. No original checkout is needed.

## License and citation

The authors have **not selected a project license**. No new license grant is
implied. Third-party notices are preserved in [LICENSES/](LICENSES/) and
[THIRD_PARTY.md](THIRD_PARTY.md). Cite the paper title above; author, venue and
persistent publication identifiers should be added when finalized.

See [release verification and limitations](docs/RELEASE_STATUS.md) before running
expensive experiments or interpreting the evidence.
