# Reading the L7 reports (hand-written)

Three runs, all on this machine under load from other agents' builds and tests (8 vCPUs; the machine could not be emptied).

- [l7.md](l7.md): the final candidate aa44c1d9 (after T2.16's encoding fix), 5 repeats over every S0 and S1 case that expects a success, from 13:10 to 13:15 UTC on 2026-10-21, load average about 41 to 45.
- [be55838a/l7.md](be55838a/l7.md): the same run on be55838a (before the fix), 09:49 to 09:54 UTC, load average about 44 to 48.
- [be55838a/l7-rerun-r21.md](be55838a/l7-rerun-r21.md): be55838a again, only what failed there, 21 repeats, 11:08 UTC, load average about 43.

| Limit | aa44c1d9, 5 repeats, 175 judged | be55838a, 5 repeats, 175 judged | be55838a, 21 repeats, the two that failed |
|---|---|---|---|
| Median core-to-reference CPU, at most 1.0 | 0.582, holds | 0.554, holds | (not judged on two scenarios) |
| Every scenario at most 1.1 | 0 above, holds | fails: `s0.accept.sum-window` 1.125, `s0.boundary.deepest` 1.201 | 1.022 and 0.989, hold |
| Import time, core at most the reference's | 1,008.4 against 1,078.1 ms, holds | fails: 1,065.7 against 974.8 ms | 1,131.2 against 1,150.8 ms, holds |
| Native library at most 1.25 times 91a00319's | 1.170, holds | 1.170, holds | – |
| Boundary overhead, message sizes, no pandas in providers | hold | hold | hold |
| Statements per scenario | 4 lines differ | the same 4 | – |

The 4 statement lines are registered refusals that stop before materialize's reads, so they issue fewer statements than the reference, which answers: `sample.num-neighbors.minus-one` (P2I-4), and `s0.boundary.date-long-field` and `s0.boundary.date-packed`.
The report counts them as a failure of limit 6 because it compares statement counts on every scenario, refusals included; they are fewer statements, never more.

Our reading: the core is within KO2-29's CPU limits.
The two crossings and the import time on be55838a did not repeat at 21 repeats, nor on the final candidate's own run; process CPU still moves under contention, so we read them as machine noise.
That is a judgement, not a measurement on a quiet machine, and it is on the packet's "not proven" list.
