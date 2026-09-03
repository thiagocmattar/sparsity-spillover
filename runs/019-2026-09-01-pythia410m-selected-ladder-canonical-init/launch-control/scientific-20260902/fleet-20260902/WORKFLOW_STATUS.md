# Run 019 terminal workflow status

Last controller update: 2026-09-03 10:10 UTC / 07:10 BRT.

All twelve scientific rows are terminally complete. Every archive matched its
remote SHA-256 before deletion, all artifacts were merged locally, the cohort
verifier returned `verified 12`, both TEAL frontiers consolidated to 20/20
points, and the focused and bootstrap test suites passed. The final RunPod
inventory contains zero GPU Pods and zero endpoints.

The last teardown balance was $83.1324. Remaining experiment ETC and required
balance are zero. Exact lifetime billing for the twelve scientific Pod IDs is
$306.5626 GPU + $2.5141 Pod disk = **$309.0767**. The only current account rate
is approximately $0.01/hour for the pre-existing retained 100 GB Standard
network volume `9luykg5yc3`; Run 019 did not attach it.

| Condition | Pod | Train | TEAL | Verify | Package | Local hash | Delete | Billed USD |
|---|---|---|---|---|---|---|---|---:|
| A0 | `fo1h6sbbohjf9w` | DONE 712/712 | DONE 10/10 | DONE | DONE | DONE | DONE | 18.4931 |
| A1-H | `c2o0tqtsu1c677` | DONE 712/712 | DONE 10/10 | DONE | DONE | DONE | DONE | 13.9294 |
| A4 OL1 k=0 | `9cayqnss01lq7q` | DONE 712/712 | N/A | DONE | DONE | DONE | DONE | 26.4129 |
| A4 OL1 k=.01 | `3o9ksu1bam0y4m` | DONE 712/712 | N/A | DONE | DONE | DONE | DONE | 27.0168 |
| A4 OL1 k=.05 | `2p3zhy9rh4if1a` | DONE 712/712 | N/A | DONE | DONE | DONE | DONE | 30.4658 |
| A4 OL1 k=.1 | `sle3lrr9oz2dhf` | DONE 712/712 | N/A | DONE | DONE | DONE | DONE | 33.5442 |
| A4 OL1 k=.5 | `tz8ih5yfjhxx6l` | DONE 712/712 | N/A | DONE | DONE | DONE | DONE | 26.4131 |
| A7 OL1 k=0 | `mjpbwge4e257ug` | DONE 712/712 | N/A | DONE | DONE | DONE | DONE | 27.7137 |
| A7 OL1 k=.01 | `dipyx36j0r0zj7` | DONE 712/712 | N/A | DONE | DONE | DONE | DONE | 28.7386 |
| A7 OL1 k=.05 | `ql1f798i6q752o` | DONE 712/712 | N/A | DONE | DONE | DONE | DONE | 29.2314 |
| A7 OL1 k=.1 | `yi8idcrkeh9etg` | DONE 712/712 | N/A | DONE | DONE | DONE | DONE | 32.2416 |
| A7 OL1 k=.5 | `rosnxzyto0ggpp` | DONE 712/712 | N/A | DONE | DONE | DONE | DONE | 14.8761 |

Terminal closeout checklist:

1. DONE — all workers reached 712 optimizer boundaries with no skipped step.
2. DONE — A0/A1-H each completed and verified ten TEAL targets.
3. DONE — all per-condition remote scientific verifiers passed.
4. DONE — 12 base and two TEAL archives were packaged.
5. DONE — every archive was retrieved and matched its remote SHA-256.
6. DONE — each exact Pod was deleted and subsequently absent from inventory.
7. DONE — scientific artifacts and retained checkpoints were merged locally.
8. DONE — terminal cohort and TEAL verification passed.
9. DONE — focused 23-test and full 183-test suites passed.
10. DONE — final control-plane audit found zero GPU Pods/endpoints.
11. DONE — all 12 obsolete local deadline guards were stopped; no Run 019
    monitor or guard process remains.
