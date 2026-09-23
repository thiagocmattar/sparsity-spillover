# Local 70M smoke (calibration only)

Four training blocks, two passes, one process per checkpoint; calibration only.

| Checkpoint | Native | Original port | opt073 | opt073 h/z skip off | Native h/z replacement | Peak allocated GiB | Min. free after capture GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| c00 | 5.9382 | 8.9891 | 8.5496 | 8.8707 | 5.6687 | 4.92 | 7.97 |
| c26 | 6.2573 | 5.7111 | 4.7063 | 7.2570 | 6.0216 | 4.92 | 7.97 |
| c24 | 6.0282 | 7.6829 | 7.4303 | 8.0618 | 5.9000 | 4.92 | 7.97 |
| c25 | 6.2329 | 7.8897 | 6.9146 | 8.5415 | 4.9632 | 4.92 | 7.97 |

Latencies are geometric-mean synchronized host milliseconds. c00=Base; c24/c25/c26=T2/Ph kappa .05/.1/.5.
All cells passed only the four-block numerical smoke. These are not final speedup or quality claims.
Source: `05_reduce.py`, with per-file hashes in the accompanying summary JSON.
