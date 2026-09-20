# Initialization-check attempt 001

All four preflights stopped on the approved initial-parameter hash check,
before constructing an optimizer or executing an optimizer boundary. The raw
JSON files retain their last status (`running`); the corresponding remote logs
record `RuntimeError: Initialization mismatch`, and pipeline exit status was 1.
They are failed calibration records, not scientific training conditions.
The complete data caches passed their required identities and were reused.
