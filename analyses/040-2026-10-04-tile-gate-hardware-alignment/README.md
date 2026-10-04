# Tile gates aligned with the retained GPU implementations

Status: source audit complete; training design remains proposed. No GPU job,
new experiment code, benchmark or manuscript edit was performed.

The user requests a tile structure whose zeros can prevent weight loads and
whose definition can carry across Pythia14M/31M/70M and other model families.
The next training comparison remains 14M T2/Ph at kappa=0/.1, measuring quality,
activation zeros and empty tiles before any inference-speed experiment.

The original audit recommends fixed contiguous groups of16 input features per
token (1x16 gate), with independent hardware execution row groups. Its source
observation remains unchanged. The user subsequently selects a stronger causal
restriction: decide a shared mask at the first token and enforce it on all
following tokens in the same fixed block. The updated proposal is8x16 with
only the first row participating in the threshold decision, as detailed in
the backlog. Exact geometry and anchor trigger await confirmation.

Both use K16 support compatible with the inspected consumers. Neither is a
measured optimal gate or proof of transferable speedup.

See [the observation](observations/001-tile-geometry-and-weight-loads.md) for
source locations, causal semantics, portability and evidence limits; the
[source identities](data/source-identity.json) preserve inspected file hashes.
The [backlog](../../research/BACKLOG.md) retains the proposed training recipe,
literal any-zero/any-below-kappa anchor trigger, unchanged Ph, backward rule,
preserved values in admitted tiles and open design choices.
