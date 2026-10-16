# Phase 2 tracker

Status values: not started, in progress, in review, approved, landed, dropped.
Each row links its evidence when landed: the landing SHA, the candidate's exact-SHA CI run and both review reports.
A package's own notes go in its own file, `tracker-<ID>.md`, so lanes never edit the same lines (design, "Merge rules"; question K4).

## Packages

| ID | Package | Lane | Status | Evidence | Notes |
|---|---|---|---|---|---|
| T2.0 | Tooling phase-aware, first checkpoint, rust-core's CI green | A | in progress | – | [tracker-T2.0.md](tracker-T2.0.md) |
| T2.0a | Design, decisions, tracker; learnings 17-22 | – | in review | Rounds below | Judging data: the locked-set file |
| T2.0b | Carried tooling rows, reference-replacement rule | A | not started | – | Waits for K1 |
| T2.0c | Update from `main`, reference rerun | ref | not started | – | Waits for T2.0b and K2's pushes |
| T2.0d | Refusal census | A | not started | – | Waits for K5; its first run (DuckDB) precedes Q2.4's answer; other backends unmeasured until T2.2 and T2.3 |
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
| T2.8c | Multi-call `predict` | D | not started | – | `main`'s 1fc5a686 |
| T2.9 | Forecasting | E | not started | – | |
| T2.10 | PQL targets | E | not started | – | |
| T2.13 | Query and option long tail | E | not started | – | Scope from Q2.4 |
| T2.12 | Schema rules | F | not started | – | `float_nan` first |
| T2.11 | Sampling options | F | not started | – | |
| T2.14a | Bridge freeze widened | A | not started | – | |
| T2.14b | Host clean-up | A | not started | – | |
| T2.14c | Bridge freeze lowered | A | not started | – | |
| T2.15 | Offline L7 | C | not started | – | Waits for Q-L7, K5 and K6 |
| T2.16 | The slice gate | A | not started | – | |
| HO2 | The new held-out set | orchestrator | not started | – | Not a landing; frozen after T2.1a |
| CI5 | Faster CI | A | not started | – | Waits for K1 |
| CI6 | Merge-queue landing | A | not started | – | Waits for K1 |

## Stop-rule counters

Integers per H20 as revised by E29: a check failing twice the same way goes to the owner; two fix rounds on one divergence go to the owner (plan section 9).

| Package | Same-way failures | Fix rounds on one divergence | Escalated |
|---|---|---|---|

## Design review rounds (T2.0a)

| Round | SHA | Claude subagent | Codex | P1 | P2 | What changed |
|---|---|---|---|---|---|---|
| 1 | 77c8c8c3 | CHANGES REQUIRED (1 P1, 10 P2, 14 P3) | CHANGES REQUIRED (2 P1, 4 P2, 2 P3) | 3 | 14 | Fix 1 (1069a5c0): the refusal census (T2.0d) and offline L7 (T2.15) added as packages and locked checks; coverage and `Request` pickles in the checks; the reference-replacement rule (T2.0b) and a second update at the gate; multi-call `predict` (T2.8c) and its vocabulary in v26; the freeze widened alone (T2.14a); HO2; the key-type matrix in T2.2 and T2.3; letter-led decision IDs; Q2.4 over 37 rules; `Q-G<n>` names; the P3s |
| 2 | 9e175ca2 | CHANGES REQUIRED (33 of 33 round 1 findings fixed; 0 P1, 4 P2, 10 P3) | CHANGES REQUIRED (0 P1, 3 P2, 2 P3) | 0 | 7 | Fix 2: L7 as the plan words it, with K6's limits fixed before implementation; the census covers rule-less and host refusals, runs first on DuckDB with other backends unmeasured, and is judging (T2.0d), as is L7 (T2.15); the freeze lowered alone (T2.14c); T2.8c completed; interim `main` differences registered at T2.0c; the held-out negative controls run for real; honest critical-path hours; `KO2-` row IDs; H24 in Q2.7; K7 |
| 3 | 0e3ec975 | CHANGES REQUIRED (17 of 19 fixed; 0 P1, 3 P2, 6 P3) | CHANGES REQUIRED (0 P1, 2 P2, 2 P3) | 0 | 5 | Fix 3: the locked-set file written whole at the lock in T2.0's format, its steps run by T2.1a's gate and its CI checks read from the candidate's run, so no later package edits it; L7's per-scenario reading asked in K6; Postgres common-input comparison built in T2.1a and run in T2.3 (H24); census hits outside the 37 handled at once (Q-C); the lock waits for Q2.4; the P3s |
| 4 | a416950e | CHANGES REQUIRED (15 of 15 fixed; 0 P1, 2 P2, 4 P3) | APPROVE (0 P1, 0 P2, 2 P3) | 0 | 2 | Fix 4: the locked-set file protected by T2.0a's landing; T2.1a's gate reads its `steps`, `ci_jobs` (by `--ci-run`) and `limits`; the case floor stated; T2.1a after T2.0a and re-estimated; stray lines from fix 3 |
| 5 | 4b114203 | CHANGES REQUIRED (6 of 8 fixed; 0 P1, 2 P2, 6 P3) | APPROVE (0 P1, 0 P2, 2 P3) | 0 | 2 | Fix 5: the nine `edit.*` cases admitted by T2.12; CI checks locked by logical name and mapped to jobs by a protected `ci-checks.yaml`; the CI-check reader's seeded faults; the owner answers on the critical path; the P3s |

## Carried rows (plan section 8)

Each row is fixed in the package named, or closed by the decision named; T2.16 takes any row no lane reached (lesson 11).

| Row | Package or decision |
|---|---|
| FR2 P3-1 `float_nan` | T2.12, first |
| G15, M4, FX-1, FX-4 | T2.14b if Q-W keeps them in Phase 2 |
| C1 | Closed: fixed at 47cdd99e (`tests/test_t132b_cell_claims.py`) |
| C2, C5, C22, C28, G4, G10, U3, U4, U13, M5, FX-6, FR P3-7 (ratchet parts) | T2.0b |
| C4 | CI5 |
| C6, C18 | T2.1a |
| C7, C11, G2 | The lane that next touches the file, else T2.16 |
| C8, C9(1), C9(2), FX-2, M7, C16 (rest) | Test-author batch |
| U14 | Closed in T2.0b's landing once both reviewers confirm no job pulls from a public registry (the CI images moved to ghcr.io, `ci.yml`), recorded as a lane A row |
| M1, M2, M3 | T2.1b |
| M9, G5 | T2.5 |
| M10 | T2.8b |
| G6, U12 | T2.2 |
| G12, G16, G17, M8 | T2.1a |
| M6 | T2.11 |
| FX-3 | T2.12 if Q-FX3 keeps it in Phase 2 |
| FR P3-8 | T2.12 |
| FR P3-2 | T2.14a |
| FR P3-7 (packet parts) | T2.16 |
| U1 | T2.13 |
| G8 | Before any live rerun; T2.16 if live evidence is used |
