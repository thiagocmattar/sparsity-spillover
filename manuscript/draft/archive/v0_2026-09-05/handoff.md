# Clean-repository handoff

## Purpose

Bootstrap a small, auditable research repository that can reproduce and extend
the paper's exact-zero accounting without importing this repository's
historical run machinery. The new repository should answer human-approved
questions about activation topology, quality, and logical product opportunity.
It must not silently choose an experiment, optimization regime, or cloud bill.

## Scientific contracts to carry over

1. Construct Pythia from a pinned architecture config with random
   initialization. Released weights constitute continuation/fine-tuning and
   must be named as such.
2. Keep the sites `a,m,h,q_pre,k_pre,q_post,k_post,v,z` exact. Pythia's `a`
   and `m` are separate parallel-branch normalization outputs. Credit QK zeros
   only after RoPE.
3. Keep topology, gate operator, gate threshold, pressure method, pressure
   sites, and pressure weight as independent config fields.
4. Pool integer counts before dividing. Keep exact zero, named near-zero mass,
   logical product opportunity, and measured runtime as different metrics.
5. For the present validation contract, evaluate all 500 MiniPile documents:
   338 complete length-2,048 sequences (692,224 tokens) and report the excluded
   1,444-token tail.
6. Define `R_model` over fused QKV, valid-causal QK, valid-causal PV, `W_o`,
   `W_1`, and `W_2`, with a declared dense LM-head denominator. Sequence length
   and workload mode are part of the metric identity.
7. Retain training-time gradient-interaction diagnostics when they may be
   needed; checkpoints cannot reconstruct them.

## Minimal layout

```text
new-repo/
  README.md
  pyproject.toml
  research/
    DEFINITIONS.md
    DATA.md
    METHODS.md
    METRICS.md
    COMPUTE.md
  src/sparsity_spillover/
    model.py          # pinned config construction and initialization identity
    sites.py          # exact GPT-NeoX ports and capture hooks
    gates.py          # one-sided/symmetric operators
    pressure.py       # naive L1 and OL1 update only
    logical.py        # integer per-operation counts and analytic reach
    diagnostics.py    # activation, weight, and gradient accumulators
    data.py           # immutable cache verification and block order
    train.py          # one explicit run loop
    evaluate.py       # full loss, activation, and logical passes
    artifacts.py      # small manifest/event/checkpoint serialization helpers
  tests/
    test_sites.py
    test_gates.py
    test_logical_counts.py
    test_aggregation.py
    test_ol1.py
    test_data_coverage.py
    test_artifact_roundtrip.py
  runs/
  analyses/
  manuscript/
```

Do not begin with a registry framework, scheduler, catalog parser, generalized
workflow engine, or central plotting package. Run-specific configs, scripts,
observations, artifacts, and figures stay in the numbered run folder. Move
code into `src/` only when a second approved run genuinely reuses it.

## First implementation slice

Implement a synthetic two-layer GPT-NeoX-shaped fixture before any real run.
The fixture should prove:

- hook placement distinguishes `q_pre` from `q_post` and `z` from post-`W_o`;
- equality survives the gates and rejected inputs have zero input gradient;
- QK/PV denominators contain exactly `T(T+1)/2` pairs per head and never count
  future-mask entries as zeros;
- per-operation integer numerators sum exactly to `R_block` and `R_model`;
- the dense head is denominator-only;
- batch/layer partitioning cannot change a pooled fraction;
- OL1 projects only negative Adam-relative interactions and respects the trust
  budget.

After those tests, reproduce one existing A0 checkpoint diagnostic locally as
a read-only compatibility target. Do not launch training merely to validate the
new repository.

## Immutable inputs

Pin these identities in `research/DATA.md`, but keep large files outside git:

- MiniPile revision `18ad1b0c701eaa0de03d3cecfdd769cbc70ffbd0`;
- tokenizer revision `7386d9a4ae45aef494a6e704910394def3037fc5`;
- training cache SHA-256
  `da82a2ea2e0080c7fd681c7a93b07d3d9ff3d5357a8640895a82d536a1eaf97c`;
- validation cache SHA-256
  `51cd758fda72f14383da30c358a895d0223c0d1d80b31455d2d842c3656d0451`.

Either copy and verify those exact cache bytes or rebuild and record a new
verified identity. Never infer cache compatibility from filenames.

## Artifact contract

Each executed attempt needs only:

- resolved `config.yaml` with immutable model/data/code identities;
- `manifest.json` with status, command, environment, coverage, and file hashes;
- append-style `events.jsonl` for optimizer boundaries and validation events;
- `diagnostics/activation_statistics.json`, `weight_statistics.json`, and
  `logical_products.json` containing integer accumulators;
- retained checkpoints explicitly selected in the approved design;
- an inventory with byte sizes and SHA-256 values.

Keep executed run records append-only. A scientific-input correction creates a
new run; an infrastructure retry creates a new attempt under the unchanged run.

## Human decision gates

Before implementing a new experiment, write and obtain confirmation for the
hypothesis, matched and varied factors, complete validation coverage, seeds,
budget, optimizer, topology, gate, pressure sites/method, checkpoints,
diagnostics, claims affected, support/refutation criteria, and interpretation
limits. After implementation and smoke tests, separately request launch
approval with exact local ETC or live cloud price/cost ceiling, storage,
transfer, monitoring, warnings, and teardown. Approval of the design is not
approval to spend compute.

## Paper-facing outputs

Create one numbered cross-run analysis for each paper comparison. It should
consume immutable manifests/diagnostics, rederive all fractions from integers,
write machine-readable figure data, generate PDF-only figures, and pair every
figure with an observation markdown file. A paper table should be generated
from the same reduced artifact as its figure.

The v0 paper bundle in this repository is the initial specification:
`manuscript/draft/main.tex`, `supplementary-results.md`, `claim-ledger.md`, and
the source-hashed figures/tables. Treat its claims as targets for reproduction,
not as evidence in the new repository.

## Open decisions for the human

- whether the next confirmation prioritizes replicate 14M seeds, matched
  tokens-per-parameter, or a longer 410M horizon;
- whether future accounting targets full-sequence prefill, cached decoding, or
  both as separately named workloads;
- which sparse kernel/layout is scientifically relevant enough to convert
  logical opportunity into a runtime experiment;
- whether A4 versus A7 should be isolated by matching pressure sites, by
  matching gates, or by a factorial design.

These alternatives answer different questions and must not be selected by the
bootstrap implementation.

