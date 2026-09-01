# GPU calibration records

No calibration has been launched. Retrieved non-evidence calibration JSON files
will be placed in this directory only after their remote bytes and SHA-256 have
been verified locally.

Each candidate runs the identical exact workload: five complete optimizer
boundaries for A0, A4-OL1 at `kappa=0.5`, and A7-OL1 at `kappa=0.5`; boundary
one is a warm-up and four boundaries drive the timing summary. The calibration
also measures complete validation, eager activation diagnostics, eager logical
diagnostics, all-parameter weight statistics, checkpoint
serialization/hash/reload, TEAL threshold calibration, and one complete TEAL
point. It is explicitly not scientific evidence.

`09_compare_calibrations.py` accepts at least two retrieved records, verifies
their initialization, schedule, and run-code identities, and emits cost versus
twelve-way makespan rows plus the nondominated set. It deliberately leaves the
GPU selection null for the human decision.
