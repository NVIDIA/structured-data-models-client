# Phase 3 tracker

One row per package; each package keeps its own `tracker-<ID>.md` (KO2-17) and writes only there.
The slip report (`phase_stats.py --forecast forecast.yaml`) gives the phase's lateness against [design.md](design.md)'s forecast at each check-in.

## Packages

| ID | Package | Lane | Stack | Candidate | State |
|---|---|---|---|---|---|
| T3.0 | Process mechanisms | A | 0 | | in review |
| T3.1 | The design | A | 0 | | in review |
| T3.2 | The tooling train | A | 1, front | | not started |
| T3.3 | The warehouse harness | H | 1 | | not started |
| T3.4 | Databricks and contract v28 | B | 2 | | not started |
| T3.5 | Views | C | 3 | | not started |
| T3.6 | Snowflake | D | 3 | | not started |
| T3.7 | Lakebase, the live runners and the phase-end reports | G | 2 | | not started |
| T3.8 | Switch-ready: the forced-route failures | E | 1 | | not started |
| T3.9 | Differences, coverage and DuckDB 2.0 | F | 2 | | not started |
| T3.10 | The switch and the switch gate | A | alone | | not started |
| T3.11 | Java binding, API and Java S0 | J | 3 | | not started |
| T3.12 | Java S1 through JDBC | J | 3 | | not started |
| T3.13 | Java on Databricks | J | alone | | not started |
| HO3 | The held-out set (not a landing) | orchestrator | – | | not started |

## Counters

Read at each check-in (learnings.md, lessons 8 and 12): round 4 of any package goes to the owner with options; a finding family blocking two rounds in a row gets a half-page rule note (KO3-3).

| Counter | Value |
|---|---|
| Landings so far / forecast | 0 / 14 |
| Judging trains beyond T3.2 (at most one catch-up) | 0 |
| Live runs (C-1 once; the end campaign, forecast twice) | 0 |
| Warehouse cases pending live (added after C-1, recorded in E-0) | 0 |
| Warehouse cases live only (neither exact replay nor the local engine can judge them; judged in E-1) | 0 |
| T3.0's landing SHA, which KO3-56's push refusal is cited by | not yet landed |
| Live failures for non-product reasons (three stop live work) | 0 |
| Packages at review round 4 or later | none |
| Finding families blocking twice | none |

## Design review rounds

| Round | SHA | Claude | Codex | Findings |
|---|---|---|---|---|
| 1 | 3c721a16 | CHANGES REQUIRED: 3 P1, 13 P2, 12 P3 | CHANGES REQUIRED: 4 P1, 2 P2 | Fixed in the round-2 commits: the offline judge (choice 2), the train's file list against T3.0's narrowed judging paths, the capture's timing and rules, a slot-consistent forecast, live evidence on the landing commit, T3.1's own checks, the report tools behind checks 2, 6 and 11, the carried items placed exactly once, the overnight decisions, Snowflake's prerequisites, Java (KO3-27 to KO3-35) |
| 2 | d6548108 | CHANGES REQUIRED: 3 P1, 6 P2, 11 P3 | CHANGES REQUIRED: 1 P1, 4 P2, 1 P3 | Fixed in fix 4: the hybrid stand-in (exact replay, local engine, live-only list) with its blind spots and draw checks, protected as a judging path; `sdm-jni` classified in T3.2's `l0_deps.py` and scanned by `l0_scan.py`; Java legs admitted per provider; handoff on the new legs (T3.3, T3.11); T3.6 after T3.5; offline warehouse L7 dropped; KO3-57 narrowed to the question's words; aarch64 cells kept on their runtimes on Graviton nodes |
| 3 | 9fcf3c27 | CHANGES REQUIRED: 1 P1, 5 P2, 6 P3 | CHANGES REQUIRED: 2 P1, 2 P2 | Fixed in fix 5: the stand-in and captures protected by trailer (`contract/protected-paths.txt`), T3.3 committing C-1's captures and T3.4 extending the stand-in; the stand-in as a provider in `sdm-testkit`, served to Python and Java harnesses by a pipe, with no L0 change; the `live_only` outcome in the tools, admitted only by the stand-in's reason codes and discharged only by E-1; one warehouse-evaluated table per derived relation; T3.6 reviewed on T3.5's approved candidate; KO3-57 mapped exactly; offline warehouse L7 by `l7.py --warehouse`; aarch64 probes install and import the wheel |
| 4 | 1c5d75ea | CHANGES REQUIRED: 3 P2, 1 P3 | APPROVE | Fixed in fix 6: T3.2 carries the stand-in's observation field (schema, `run_python.py`, `observation.rs`) and python-core's wiring to the forwarding provider; the captures-only commit after the final run is the one exception to the rerun rule, held by `capture_check.py`; mixed key types answered on the local engine and flagged B4, live-only only on a difference; the admissible codes named exactly |

## The checking tools' dry run

| Run | Base | Result |
|---|---|---|
| Draft (2026-10-22), from T3.0's topic branch, no runners, the first 21 kinds | Phase 2's last landing | 20 of 21 kinds with gaps: see design.md, "The checking tools' dry run" |
| At lock, on T3.0's tree, all 31 kinds and the reports list, with `--python`, `--rust` and `--gate 3a` | | |
