# Reading the L7 reports (hand-written)

[l7.md](l7.md) is the full run: 5 repeats over every S0 and S1 case that expects a success, measured from 09:49 to 09:54 UTC on 2026-10-21 under a load average of about 44 to 48 on 8 vCPUs (other agents' builds and test runs; the machine could not be emptied).
[l7-rerun-r21.md](l7-rerun-r21.md) reran only what failed there, at 21 repeats at 11:08 UTC, under a load average of about 43.

| Limit | 5 repeats, all 175 judged scenarios | 21 repeats, the two that failed |
|---|---|---|
| Median core-to-reference CPU, at most 1.0 | 0.554, holds | (not judged on two scenarios) |
| Every scenario at most 1.1 | fails: `s0.accept.sum-window` 1.125, `s0.boundary.deepest` 1.201 | 1.022 and 0.989, hold |
| Import time, core at most the reference's | fails: 1,065.7 against 974.8 ms | 1,131.2 against 1,150.8 ms, holds |
| Native library at most 1.25 times 91a00319's | 1.170, holds | – |
| Boundary overhead, message sizes, no pandas in providers | hold | hold |
| Statements per scenario | 4 lines differ: registered refusals that stop before materialize's reads (`sample.num-neighbors.minus-one`, P2I-4; `s0.boundary.date-long-field` and `s0.boundary.date-packed`) | – |

Our reading: process CPU (`time.process_time`) still moves under heavy contention (caches, frequency), and both scenarios that crossed 1.1 sit within run-to-run noise at 21 repeats; we read the 5-repeat failures as machine noise, not a regression.
That is a judgement, not a measurement on a quiet machine: a rerun on an idle machine would settle it, and it is on the packet's "not proven" list.
