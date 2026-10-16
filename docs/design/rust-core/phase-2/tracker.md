# Phase 2 tracker

Status values: not started, in progress, approved, landed, dropped.
Each row links its evidence when landed: the landing SHA, the candidate's exact-SHA CI run and both review reports.
A package's own notes go in its own file, `tracker-<ID>.md`, so lanes never edit the same lines (design, "Merge rules"; question K4).

## Packages

| ID | Package | Lane | Status | Evidence | Notes |
|---|---|---|---|---|---|
| T2.0 | Tooling phase-aware, first checkpoint, rust-core's CI green | A | in progress | – | [tracker-T2.0.md](tracker-T2.0.md) |
| T2.0a | Design, decisions, tracker; learnings 17-22 | – | in review | Rounds below | |
| T2.0b | Carried tooling rows | A | not started | – | Waits for K1 |
| T2.0c | Update from `main`, reference rerun | ref | not started | – | Waits for K2's pushes |
| T2.1a | The S1 judging path | A | not started | – | |
| T2.1b | Contract v26 | A | not started | – | |
| T2.2 | SQLite | B | not started | – | |
| T2.3 | Postgres | B | not started | – | |
| T2.4 | Full L4 and L5 | B | not started | – | |
| T2.5 | Conformance kit | C | not started | – | |
| T2.6 | In-memory fallback | C | not started | – | |
| T2.7 | Non-SQL provider and swapped data layer | C | not started | – | |
| T2.8a | Ranking, materialize side | D | not started | – | |
| T2.8b | Ranking, predict side | D | not started | – | |
| T2.9 | Forecasting | E | not started | – | |
| T2.10 | PQL targets | E | not started | – | |
| T2.11 | Sampling options | F | not started | – | |
| T2.12 | Schema rules | F | not started | – | `float_nan` first |
| T2.13 | Query and option long tail | F | not started | – | Scope from Q2.4 |
| T2.14 | Host clean-up | A | not started | – | |
| T2.16 | The slice gate | A | not started | – | |
| CI5 | Faster CI | A | not started | – | Waits for K1 |
| CI6 | Merge-queue landing | A | not started | – | Waits for K1 |

## Stop-rule counters

Integers per H20 as revised by E29: a check failing twice the same way goes to the owner; two fix rounds on one divergence go to the owner (plan section 9).

| Package | Same-way failures | Fix rounds on one divergence | Escalated |
|---|---|---|---|

## Design review rounds (T2.0a)

| Round | SHA | Claude subagent | Codex | P1 | P2 | What changed |
|---|---|---|---|---|---|---|

## Carried rows (plan section 8)

Each row is fixed in the package named, or closed by the decision named.

| Row | Package or decision |
|---|---|
| FR2 P3-1 `float_nan` | T2.12, first |
| G15, M4, FX-1, FX-4 | T2.14 if Q-W keeps them in Phase 2 |
| C1 | Closed: fixed at 47cdd99e (`tests/test_t132b_cell_claims.py`) |
| C2, C5, C22, C28, G4, G10, U3, U4, U13, M5, FX-6, FR P3-7 (ratchet parts) | T2.0b |
| C4 | CI5 |
| C6, C18 | T2.1a |
| C7, C11, G2 | The lane that next touches the file |
| C8, C9(1), C9(2), FX-2, M7, C16 (rest) | Test-author batch |
| U14 | To close by recorded decision: the CI images moved to ghcr.io (`ci.yml`), so registry rate limits no longer apply |
| M1, M2, M3 | T2.1b |
| M9, G5 | T2.5 |
| M10 | T2.8b |
| G6, U12 | T2.2 |
| G12, G16, G17, M8 | T2.1a |
| M6 | T2.11 |
| FX-3 | T2.12 if Q-FX3 keeps it in Phase 2 |
| FR P3-8 | T2.12 |
| FR P3-2 | T2.14 |
| FR P3-7 (packet parts) | T2.16 |
| U1 | T2.13 |
| G8 | Before any live rerun; T2.16 if live evidence is used |
