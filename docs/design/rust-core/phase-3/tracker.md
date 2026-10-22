# Phase 3 tracker

One row per package; each package keeps its own `tracker-<ID>.md` (KO2-17) and writes only there.
The slip report (`phase_stats.py --forecast forecast.yaml`) gives the phase's lateness against [design.md](design.md)'s forecast at each check-in.

## Packages

| ID | Package | Lane | Stack | Candidate | State |
|---|---|---|---|---|---|
| T3.0 | Process mechanisms | A | alone | | in review |
| T3.1 | The design | A | alone | | in review |
| T3.2 | The tooling train | A | 1, front | | not started |
| T3.3 | Databricks and contract v28 | B | 2 | | not started |
| T3.4 | Views | C | 3 | | not started |
| T3.5 | Snowflake | D | 3 | | not started |
| T3.6 | Lakebase and the live runners | G | 2 | | not started |
| T3.7 | Switch-ready: the forced-route failures | E | 1 | | not started |
| T3.8 | Differences, coverage and DuckDB 2.0 | F | 1 | | not started |
| T3.9 | The switch and the switch gate | A | alone | | not started |
| HO3 | The held-out set (not a landing) | orchestrator | – | | not started |

## Counters

Read at each check-in (learnings.md, lessons 8 and 12): round 4 of any package goes to the owner with options; a finding family blocking two rounds in a row gets a half-page rule note (KO3-3).

| Counter | Value |
|---|---|
| Landings so far / forecast | 0 / 10 |
| Judging trains beyond T3.2 (at most one catch-up) | 0 |
| Live campaigns run (two planned: the early capture, the end campaign) | 0 |
| Live failures for non-product reasons (three stop live work) | 0 |
| Packages at review round 4 or later | none |
| Finding families blocking twice | none |

## Design review rounds

| Round | SHA | Claude | Codex | Findings |
|---|---|---|---|---|

## The checking tools' dry run

| Run | Base | Result |
|---|---|---|
| Draft (2026-10-22), from T3.0's topic branch, no runners | Phase 2's last landing | 20 of 21 kinds with gaps: see design.md, "The checking tools' dry run" |
| At lock, on the landed T3.0, with `--python`, `--rust` and `--gate 3a` | | |
