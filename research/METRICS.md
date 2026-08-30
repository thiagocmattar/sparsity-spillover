# Metrics and Artifact Contract

## Aggregation rule

For every fraction, save integer numerators and denominators, pool counts first,
then divide. Record coverage: source run/checkpoint, validation cache, documents,
sequences, input tokens, layers, seeds, and excluded tail.

## Training events

Each logged optimizer boundary should include:

- step, input tokens seen, elapsed and step wall time, learning rate;
- task loss;
- validation loss and full coverage fields when evaluated;
- global task-gradient norm before/after clipping and clipping flag;
- global and MLP weight norms when requested;
- pressure loss, weight, weighted pressure loss, and monitoring-only augmented
  loss when pressure is active;
- task/pressure gradient norms, ratio, dot, cosine, and conflict flag. For naive
  L1 these are the separate unweighted gradients; for OL1 the task gradient is
  the globally clipped gradient actually consumed by AdamW while the pressure
  gradient is unclipped. The AdamW pre/post-clip norms are logged separately;
- OL1 task/pressure direction norms, projection flag, before/after dot/cosine,
  raw ratio, trust scale, and final ratio;
- activation exact/near-zero counts and RMS for every requested site/layer when
  training-time capture was approved;
- tokens/second and relevant peak memory.

Do not store only the last batch's activation percentages as if they were a
validation diagnostic.

## Activation statistics

For every requested `site.layer_N`, store:

- total, finite, nonfinite, exact-zero count;
- hit count for every named near-zero threshold;
- sum, sum of squares, absolute sum;
- mean, RMS, L2 norm, and mean absolute value derived from those accumulators.

Pool a site-level RMS as:

```text
rms_site = sqrt(sum_layer(sum_squares_layer) / sum_layer(finite_count_layer))
```

## Weight statistics

Record each included parameter name, layer, role, element count, finite count,
and L2 norm. The standard Pythia roles are fused QKV, attention output, MLP W1,
MLP W2, embeddings/LM head, normalization, and bias. Site `a` feeds fused QKV,
`m` feeds MLP W1, and `h` feeds MLP W2; Q/K/V use the fused projection and must
not be presented as separate parameter tensors unless an explicit tested slice
definition is added. If pooling tensors, sum squared norms and take one square
root. Site `z` feeds the attention output projection. State whether biases and
normalization parameters are excluded.

## Logical-product metrics

Measure actual operands for these Pythia block operations:

1. fused QKV projection;
2. causal QK score products;
3. causal probability-value products;
4. attention output projection;
5. MLP W1/up projection;
6. MLP W2/down projection.

Exclude future causal positions from QK/PV denominators. Save per-operation zero
product counts and product counts.

```text
R_block = sum(block zero-product counts) / sum(block product counts)

R_model = sum(block zero-product counts)
          / (sum(block product counts) + declared dense LM-head products)
```

Measured `R_block` and `R_model` include actual zero operands in all six
operations, including naturally occurring zeros outside selected intervention
sites.

### Architecture ceiling

For a topology, architecture, and full-sequence uncached workload, define the
all-zero reachable operations:

```text
a                 -> fused QKV projection
m                 -> MLP W1
h                 -> MLP W2
q_pre/q_post or
k_pre/k_post      -> causal QK products (credited once)
v                 -> causal PV products and, by V=0 => C=0, output projection
z                 -> attention output projection
```

`R_model_max` is the reachable product count divided by the same block plus
dense-LM-head denominator as `R_model`. It is analytic and topology-conditioned;
it does not depend on a checkpoint's observed activation values. PRE and POST
have the same all-zero ceiling even though their partial-mask behavior differs.

Artifacts store count-derived fractions in `[0,1]`. Manuscript prose and figures
may display `100 * fraction` as a percentage, with the unit explicit.

For Pythia with FFN width `d_f`, one full-sequence block denominator is:

```text
T * (4*d^2 + 2*d*d_f) + d*T*(T+1)
```

The model denominator multiplies the block count by the number of layers and
adds `T*d*V` dense LM-head products. `ceilings.py` implements the draft
definition and exposes its integer numerator and denominator.

When reporting the ceiling, save topology ID/active sites, `L,d,d_f,T,V`, valid
causal-pair count, per-operation dense counts, reachable operations, reachable
numerator, block/LM-head/model denominators, fraction, and derived percentage.
Do not reconstruct the value later from a rounded table.

The manuscript also defines `U_arch = R_model / R_model_max` for a nonzero
ceiling. Because measured `R_model` includes all actual zero operands while the
ceiling is selected-site reach, `U_arch` is a normalization ratio rather than a
guaranteed `[0,1]` utilization. Emit it only with that numerator caveat.

## Run folder artifacts

An executed attempt lives under `artifacts/attempts/NNN-<timestamp>/` and uses:

| File | Purpose |
| --- | --- |
| `config.yaml` | Exact resolved scientific and operational inputs |
| `manifest.json` | Status, command, code/environment identity, data/model identity, coverage, checkpoint references |
| `events.jsonl` | Append-style train/validation/progress events |
| `metrics.json` | Terminal scalar metrics and complete counters |
| `predictions.jsonl` | Optional per-example or per-setting outputs; clipping rows may live here |
| `diagnostics/` | Named structured post-hoc artifacts |
| `checkpoints/final/` | Final checkpoint when retention was approved |
| `transfer_inventory.json` | Relative path, byte count, and SHA-256 for files copied from cloud |

The manifest starts as `running`; terminal success is published only after
required artifacts are durable. Failures remain `failed` and are not rewritten.

### Minimum manifest identity

- run/attempt ID, status, start/finish timestamps, command;
- Git commit and dirty-state disclosure;
- Python, Torch, Transformers, CUDA, device, precision;
- full config SHA-256;
- model architecture/revision/initialization and initial-parameter hash;
- dataset/tokenizer revisions and cache hashes;
- model seed, data-order seed, schedule hash;
- completed steps and input tokens;
- validation documents/sequences/tokens/excluded tail;
- pressure, topology, and gate settings;
- checkpoint path/hash when retained.

## `predictions.jsonl`

Do not manufacture token-level predictions for a pretraining run solely to
satisfy a schema. Use the file when there is a natural row-level output, such as
one clipping threshold, source, or evaluated example. Otherwise create an empty
file or omit it and say so in the manifest. The filename remains part of the
terminology because diagnostics often produce natural per-setting rows.

## Evidence labels

- `valid` — the attempt and required diagnostics cover the approved question.
- `provisional` — a precisely named limitation restricts use.
- `invalid` — the artifact cannot support the intended comparison.

An unfavorable result may be valid. A successful process may still be
scientifically invalid.
