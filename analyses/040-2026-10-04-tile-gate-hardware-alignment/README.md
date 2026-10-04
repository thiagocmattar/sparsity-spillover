# Tile gates aligned with the retained GPU implementations

Status: source audit complete; training design remains proposed. No GPU job,
new experiment code, benchmark or manuscript edit was performed.

The user requests a tile structure whose zeros can prevent weight loads and
whose definition can carry across Pythia14M/31M/70M and other model families.
The next training comparison remains 14M T2/Ph at kappa=0/.1, measuring quality,
activation zeros and empty tiles before any inference-speed experiment.

**Recommendation: fixed contiguous groups of 16 input features per token
(1x16 gate), with independent hardware execution groups of token rows.** This
is a format compatible with existing K16 consumers, not a claim that this gate
or one kernel configuration is optimal or already faster at every scale.

See [the observation](observations/001-tile-geometry-and-weight-loads.md) for
source locations, causal semantics, portability and evidence limits; the
[source identities](data/source-identity.json) preserve inspected file hashes.
The [backlog](../../research/BACKLOG.md) retains the proposed training recipe,
literal any-zero/any-below-kappa trigger, unchanged Ph and open design choices.
