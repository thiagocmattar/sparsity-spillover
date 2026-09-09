# O003: Five retained historical h-only controls

Question: Do the five omitted A4 checkpoints supply usable pressure-placement
evidence, and do they implement the proposed fixed-coefficient design?

Method/source: `03_audit_history.py` exactly reconstructs Analysis009 using its
read-only builder, original configurations, manifests, counts and inherited
capture code. It matches all five Run 029 c31-c35 checkpoint identities and
verifies their retained source hashes/bytes. Windows extended paths are used
only for IO where a retained path exceeds260 characters. The result is
`historical-audit.json`, including original weight hashes and counterpart IDs.

Coverage: all five thresholds0/.01/.05/.1/.5, same14M initialization, order,
712 updates,1493172224 input tokens and complete338-block validation. This
audit pairs FP16 diagnostic loss with logical counts, rather than substituting
Analysis009's separately retained terminal training-validation loss deltas.

Result: Run012 actually pressured only h, despite stale metadata declaring four
sites. Run015 realizes all24 site-layer targets. Four-site minus h-only loss
differences are+.242526809,+.253796352,+.294177376,+.319554873,+.315322542;
sparsity differences are-.599370804,+.000905848,+1.420816110,+2.050795511,
+2.486058330 percentage points. Each old h term changes coefficient1/6 to1/24.

Caption/legend: no new figure; the manuscript's kernel appendix describes the
historical cohort separately, with signs always four-site minus h-only at the
same threshold. Main training and runtime cohorts remain30, not35.

Caveats: this is a valid historical equal-target-mean objective comparison,
not fixed-coefficient N4/N7 by P4/P7. It does not identify placement alone or
provide independent training replications. Original runs are unchanged.
