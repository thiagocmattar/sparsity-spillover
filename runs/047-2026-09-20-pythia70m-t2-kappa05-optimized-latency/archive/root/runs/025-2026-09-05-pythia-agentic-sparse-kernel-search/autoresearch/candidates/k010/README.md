# K010: frozen Pythia-410M flagged-z policy

K010 applies K004's exact flagged zero-tile kernel only to the attention-output
`z` projection in transformer layers 2 through 20. The fixed layer range was
selected from the A4-high/A7-high development occupancy measurements: edge
layers had substantially less executable zero-tile mass and made all-layer
dispatch unprofitable.

This is a development candidate, not held-out evidence. It uses no checkpoint
identity or runtime density dispatch, changes no weights or activations, and
leaves QK/PV attention, all other linear sites, and the LM head dense.
