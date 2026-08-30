# Infrastructure launch attempt 001

At 2026-08-29T21:09:22Z, the approved launcher stopped before creating a
scientific attempt or Python training process. Windows PowerShell
`Start-Process` could not enumerate the inherited environment because it
contained both `Path` and `PATH`, which collide under PowerShell's
case-insensitive dictionary handling.

The empty cohort logs and initial launch provenance are retained. The retry
replaces only the detached-process mechanism with a run-local Python helper
that passes a normalized environment to `subprocess.Popen`. The scientific
configuration, conditions, schedule, initialization, data, diagnostics, and
artifact contract are unchanged. The updated run-code identity is recorded in
the approved launch packet and checked by the trainer before any attempt is
created.
