# Phase 2 tracker

Status values: not started, in progress, in review, approved, landed, dropped.
Each row links its evidence when landed: the landing SHA, the candidate's exact-SHA CI run and both review reports.
A package's own notes go in its own file, `tracker-<ID>.md`, so lanes never edit the same lines (design, "Merge rules"; question K4).

## Packages

| ID | Package | Lane | Status | Evidence | Notes |
|---|---|---|---|---|---|
| T2.0 | Tooling phase-aware, first checkpoint, rust-core's CI green | A | landed | 292f8d8a; merged into `rust-core` as 3b81df85 (KO2-6) | [tracker-T2.0.md](tracker-T2.0.md) |
| T2.0e | Gate rules for `main` first: the reference approval row, the bridge freeze's `main` reason, a merge of `main` landing | A | landed | 912d9051 | Split out of T2.0b (KO2-3, KO2-15) |
| T2.0a | Design, decisions, tracker, locked-set file; learnings 17-22 | – | in review | Rounds below | Revised with the owner's answers (fix 12) |
| T2.0f | The locked-set reader for Phase 2, check 7's `main` ancestry, the `<case>@<database>` keys | H | landed | b8f704e3 | Lands after T2.0e and before T2.0a, so the lock reads with the checkpoint's reader |
| T2.0c | `main` 327fda03 merged, the reference re-recorded at 6298913c | B | in progress | – | Lands from lane B after T2.0e and T2.0a (orchestrator's decision) |
| T2.0d | Refusal census | H | not started | – | First run on DuckDB |
| T2.0b | Carried tooling rows | H | not started | – | After T2.0e (both change `ratchet.py`) |
| T2.1a | The S1 judging path, all cases on three databases | A | not started | – | |
| T2.1b | Contract v26, with the table source | A | not started | – | |
| T2.2 | SQLite | B | not started | – | |
| T2.2b | SQLite CI | B | not started | – | Judging, alone, right after T2.2 |
| T2.3 | Postgres | B | not started | – | Rust host by replay (KO2-20) |
| T2.3b | Postgres CI | B | not started | – | Judging, alone, right after T2.3 |
| T2.4 | Full L4 and L5 | B | not started | – | |
| T2.5 | Conformance kit | C | not started | – | |
| T2.6 | In-memory fallback | C | not started | – | |
| T2.7 | Non-SQL provider and swapped data layer | C | not started | – | |
| T2.8a | Ranking, materialize side | D | not started | – | |
| T2.8b | Ranking, predict side | D | not started | – | |
| T2.8c | Multi-call `predict` | G | not started | – | `main`'s 1fc5a686; 6–8 h; first wave |
| T2.9 | Forecasting | E | not started | – | |
| T2.10 | PQL targets | E | not started | – | |
| T2.13 | Query and option long tail | D | not started | – | |
| T2.12 | Schema rules | F | not started | – | `float_nan` first; lands after T2.3 |
| T2.11 | Sampling options | F | not started | – | |
| T2.17 | Exact decimals | D | not started | – | KO2-35; first wave |
| T2.18 | Any column name | E | not started | – | KO2-36; first wave |
| T2.19 | Two-path sampling | F | in progress | – | `main`'s eb6c68eb; first wave; [tracker-T2.19.md](tracker-T2.19.md) |
| T2.14a | Bridge freeze widened | A | not started | – | |
| T2.14b | Host clean-up | A | not started | – | |
| T2.14c | Bridge freeze lowered | A | not started | – | |
| T2.15 | Offline L7 | C | not started | – | |
| T2.16 | The slice gate | A | not started | – | |
| HO2 | The new held-out set | orchestrator | not started | – | Not a landing; frozen after T2.1a |
| CI5 | Faster CI | H | not started | – | |
| CI6 | Merge-queue landing | H | not started | – | |
| CI7 | aarch64 wheels and runtime cells | H | not started | – | KO2-38, KO2-40 |
| CI8 | cargo-deny exception for `zip` | C | landed | 8bcc85e4 | KO2-49; lands alone, first |

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
| 6 | 1a983d25 | CHANGES REQUIRED (8 of 10 fixed; 0 P1, 2 P2, 2 P3) | APPROVE (0 P1, 0 P2, 1 P3) | 0 | 2 | Fix 6: the locked set in T2.0's format (numbered `checks` over gate `steps`), all 52 S1 cases in it; CI checks as job-name patterns fixed now, no mapping file; skipped jobs fail; T2.4 judging for its CI jobs |
| 7 | b4a78278 | CHANGES REQUIRED (3 of 5 fixed; 0 P1, 3 P2, 3 P3) | CHANGES REQUIRED (0 P1, 3 P2) | 0 | 6 | Fix 7: check 9 outside the gate; the `inventory` step and the `every` check (12) as T2.0's reader needs; CI patterns against `ci.yml`'s real job names; the `ci` step resolves an empty landing to its tested candidate's run as E131 does; per-check CI failures; the edit cases' interim gate retired |
| 8 | 6ae9a8c7 | CHANGES REQUIRED (8 of 9 fixed; 0 P1, 2 P2, 4 P3) | APPROVE (0 P1, 0 P2, 1 P3) | 0 | 2 | Fix 8: Postgres in the judging CI jobs (T2.3 part 1, alone, never skipped); `--ci-run` through `final_gate.sh` and `phase_end.sh`; moved landings; check numbers of the CI patterns; the edit gate's retirement alone; the locked file's `inventory`, `reference` and `every` |
| 9 | e852d2b8 | CHANGES REQUIRED (5 of 7 fixed; 0 P1, 2 P2, 4 P3) | CHANGES REQUIRED (0 P1, 1 P2, 3 P3) | 0 | 3 | Fix 9: the kit run against the Python providers in `contract-python (*)`, with seeded faults per host; the Postgres CI service moved into T2.1a with `SDM_REQUIRE_POSTGRES`; the `pyo3` tree check under `rust-test*`; the interim edit gate retired only with shown replacements; the landing count |
| 10 | 6dcfcecf | CHANGES REQUIRED (8 of 10 fixed; 0 P1, 1 P2, 3 P3) | APPROVE (0 P1, 0 P2, 1 P3) | 0 | 1 | Fix 10: the reference's and the core's SQLite and Postgres drivers installed in T2.1a's judging landing with the service, and a seeded unreachable-service fault; no service in `cross-host*`; the kit's two sides share one fixture and one answers file; check 5 names the jobs that run each side |
| 11 | 6ba078b4 | APPROVE (0 P1, 0 P2, 3 P3) | APPROVE (0 P1, 0 P2, 1 P3) | 0 | 0 | Both approve 6ba078b4. Closure fixes: check 5's single Python-kit cell justified; the Postgres-required tests named; the kit fixture's path. Then one closure check |
| 12 | f6eb5f05 | – | APPROVE (closure check) | 0 | 0 | Fix 11's P3s confirmed; the design approved |
| 13 | f9324e51 | CHANGES REQUIRED (0 P1, 7 P2, 11 P3) | CHANGES REQUIRED (0 P1, 3 P2, 2 P3) | 0 | 10 | Fix 13, below |
| 14 | f10c37ca | CHANGES REQUIRED (0 P1, 3 P2, 7 P3) | CHANGES REQUIRED (0 P1, 3 P2, 1 P3) | 0 | 6 | Fix 14, below |
| 15 | 79ca161d | CHANGES REQUIRED (0 P1, 1 P2, 6 P3) | CHANGES REQUIRED (0 P1, 2 P2, 1 P3) | 0 | 2 | Fix 15, below |
| 16 | 782c528d | CHANGES REQUIRED (0 P1, 1 P2, 3 P3) | APPROVE (0 P1, 0 P2, 2 P3) | 0 | 1 | Fix 16, below |
| 17 | 2dabd79a | CHANGES REQUIRED (0 P1, 2 P2, 1 P3) | APPROVE (no findings) | 0 | 2 | Fix 17, below |
| 18 | ff6ac631 | CHANGES REQUIRED (0 P1, 1 P2, 1 P3) | CHANGES REQUIRED (0 P1, 1 P2) | 0 | 1 | Fix 18, below |
| 19 | 36a3dde2 | CHANGES REQUIRED (0 P1, 1 P2, 3 P3) | APPROVE (no findings) | 0 | 1 | Fix 19, below |
| 20 | 14e7c206 | APPROVE (0 P1, 0 P2, 3 P3) | APPROVE (no findings) | 0 | 0 | Both approve 14e7c206; fix 20 takes the three P3s, then one closure check |

## Carried rows (plan section 8)

Each row is fixed in the package named, or closed by the decision named; T2.16 takes any row no lane reached (lesson 11).

| Row | Package or decision |
|---|---|
| FR2 P3-1 `float_nan` | T2.12, first |
| G15, M4, FX-1, FX-4 | T2.14b (KO2-32) |
| C1 | Closed: fixed at 47cdd99e (`tests/test_t132b_cell_claims.py`) |
| C2, C5, C22, C28, G4, G10, U3, U4, U13, M5, FX-6, FR P3-7 (ratchet parts) | T2.0b |
| C4 | CI5 |
| C6, C18 | T2.1a |
| C7, C11, G2 | The lane that next touches the file, else T2.16 |
| C8, C9(1), C9(2), FX-2, M7, C16 (rest) | Test-author batch |
| U14 | Closed in T2.0b's landing once both reviewers confirm no job pulls from a public registry (the CI images moved to ghcr.io, `ci.yml`), recorded in `tracker-T2.0b.md` (KO2-14) |
| M1, M2, M3 | T2.1b |
| M9, G5 | T2.5 |
| M10 | T2.8b |
| G6, U12 | T2.2 |
| G12, G16, G17, M8 | T2.1a |
| M6 | T2.11 |
| FX-3 | Phase 4 (KO2-31) |
| FR P3-8 | T2.12 |
| FR P3-2 | T2.14a |
| FR P3-7 (packet parts) | T2.16 |
| U1 | T2.13 |
| G8 | Before any live rerun; T2.16 if live evidence is used |

## Revisions after approval (T2.0a)

Each revision is reviewed by both reviewers (KO2-9) before T2.0a lands.

| Fix | What changed | Why |
|---|---|---|
| 12 | decisions.md: the owner's answers of 2026-10-16 as KO2-1 to KO2-44 (KO2-20, Q2.2, owner to confirm), the reference approval format T2.0e reads, and only owner rows from now on (KO2-14). design.md: `main` first (T2.0e, T2.0c); two reviewers every round, checkpoints every 4 or 5 landings, a case lands with its code by its test author; every case on all three databases in per-database CI jobs; new packages T2.17 (exact decimals), T2.18 (any column name), T2.19 (two-path sampling, early) and CI7 (aarch64 wheels, runtime cells); T2.8c resized and unchained from T2.8b; the table source in v26 (choice 12); the Phase 5 release note; lanes G and H; critical path about 49 to 50 hours, 34 landings; question Q-I. locked-set.yaml written whole, with per-database job patterns, aarch64 and the runtime cells. migration-plan.md: `main` reaches the epic only through a phase branch (KO2-7) and the floors of KO2-37. The handbook: two reviewers on confirmations (KO2-9) | The owner's answers to the kickoff questions and to the six audits of `main`'s 20 commits (KO2-1 to KO2-44) |
| 13 | All 10 P2s of round 13. The table source's carrier stated: `Relation` replaces the schema's `source: String` and each request's `table: String` (Claude P2-1). KO2-39 and KO2-22 limited to what the owner wrote; check 2 met by a later owner answer quoting the census counts (Claude P2-2, P2-3). The first wave (T2.17, T2.18, T2.8c, T2.19) right after T2.1b at the owner's request, DuckDB legs gating their landings and the SQLite and Postgres legs exits of T2.2 and T2.3 (Codex P2-2, Claude P2-4). Reference lines for SQLite and Postgres keyed `<case>@<database>` (Claude P2-5). T2.0c from lane B; reference rows read only at the checkpoint, so T2.0a lands before T2.0c (Claude P2-6). New package T2.0f: the reader for the locked set, landed before the lock, and check 7's `main` ancestry check (Codex P2-3, Claude P2-7). The type map uses each engine's own type; L2 and L5 separated; Q-L5 asks the owner how three-driver identity reads where an engine can't store a type (Codex P2-1). P3s taken: the protected-path wording, T2.19 out of the interim refusals, migration-plan's reviewer row, cell and job names held to the locked set, `rust-test` in the timings, `.gitlab-ci.yml` left, KO2-10's and KO2-33's wording, T2.0a on the path, the kit under `contract/fixtures/provider-kit/`, decisions.md as judging data (Claude P3-1 to P3-11, Codex P3-1, P3-2); none skipped. Owner rows KO2-20 (approved), KO2-45 (Q-I) and KO2-46 (simplicity and portable interfaces); both reviewer briefs check KO2-46. Under KO2-46, cut from the table source: `columns`, `filter`, `SqlText`, the `ColumnFrom` enum, the `derived_relations` capability and the `FailureClass` field, since Phase 2 would only define and refuse them, a Rust enum with data is awkward to mirror in Java, and Phase 3 adds them as optional fields; kept: `Relation { source: Vec<String> }`, the one field whose type must change, plain data from any host. Critical path about 47 to 48 hours, 35 landings | Round 13's findings; the owner's later answers and design principle |
| 14 | All 6 P2s of round 14. T2.0f also teaches the observation schema, `compare.py` and the ratchet's removed-case and baseline rules the `<case>@sqlite` and `<case>@postgres` keys, with seeded faults, before T2.1a records any (Claude P2-1, Codex P2-1; orchestrator's decision); T2.0f grows to about 5 h and leads the critical path. KO2-25's parenthetical dropped and choice 1's failure class removed, so v26 adds only `Relation { source: Vec<String> }` (Claude P2-2, Codex P2-3). T2.0e's exit names the seeded fault of a self-written approval row, already in its tests at 6648a1e8 (Claude P2-3). Q-L5's recommendation is the design; the alternative is only a possible later package, since a SQLite column declared `DECIMAL(38,10)` already rounds what it stores (Codex P2-2). P3s taken: T2.17 moved to lane D so the first wave uses four lanes (Codex P3-1); the reference-row format loosened to the word and the full SHA, as T2.0e's tests use (P3-1); T2.0a doesn't wait for Q-L5, and coverage holds no cell for a type an engine can't store (P3-2); the ancestry check only for halves mapping check 7 (P3-3); T2.2 starts when the first slot frees (P3-4); "packages" in the risks (P3-5); the landing rate restated at about 22 a day with the docs landings (P3-6); two- and three-part names rendered with goldens, none or more than three refused by name (P3-7); none skipped. Critical path about 49 to 50 hours | Round 14's findings |
| 15 | Both P2s of round 15. T2.0f's baseline rule keyed on core observation: an admitted acceptance case needs a baseline on each database the core is observed on, with the control that a reference line on a database with no core observation passes and the fault that a core observation with no baseline fails by name (Claude P2-1, Codex P2-1; orchestrator's decision). Each dialect declares how many name parts it accepts (DuckDB and Postgres 1 to 3, SQLite 1 or 2), with goldens only for those and a named refusal with a negative vector beyond them; `Relation` itself unchanged (Codex P2-2; orchestrator's decision). P3s taken: T2.0e lands before T2.0f (Claude P3-6); E's packages serial and T2.12 landing after T2.3, the product span now about 22 to 23 h and the critical path about 50 to 51 h (Codex P3-1, Claude P3-1); Q-L5 reworded, the value kept and only the logical type lost, with costs both ways (P3-2); the ancestry check as a `main` step that the locked set lists under check 7 (P3-3); the unstorable-type list derived from the protected type map, with a seeded fault (P3-4); SQLite decimals declared `TEXT`, and a close-keys vector in T2.1a (P3-5); none skipped | Round 15's findings |
| 16 | Claude P2-1: until T2.2 and T2.3 land, the SQLite and Postgres jobs T2.1a creates run the reference side on their database and never pass empty; T2.1a's exit names the jobs and legs that exist on its run, and T2.2 and T2.3 each show their database's patterns resolve with the core legs (orchestrator's decision). P3s taken: SQLite's first part is the database (`["main", "t"]`) and Postgres's three-part form names the connected database or is refused (Codex P3-1, Claude P3-1); decimals stored as scale-normalized text, with an equal-across-scales vector in T2.1a and T2.2's matrix (Claude P3-2); the slot schedule with T2.12's Postgres leg after T2.3 and E's work at its real 7.5 h, still about 22.5 h (Codex P3-2, Claude P3-3); tracker order notes for T2.12 and T2.0f; none skipped | Round 16's findings |
| 17 | Claude P2-1: T2.2 and T2.3 create `cross-host (sqlite)`, `cross-host (postgres)` and `rust-test`'s SQLite and Postgres passes; T2.1a's exit covers only the jobs it creates; the locked patterns unchanged. Claude P2-2: Postgres accepts 1 or 2 name parts in Phase 2, three refused by name, with no connected-database input (KO2-46). Claude P3-1: one function defines a decimal's canonical text for the loader and the key binders, with `100.00`/`100` and `-0.00`/`0` in the vector. All orchestrator's decisions; nothing else touched | Round 17's findings |
| 18 | Both reviewers' P2: the SQLite and Postgres CI changes are split out as T2.2b and T2.3b, judging landings alone right after their product parts, as T2.12's retirement does; the product parts' own gate is the local three-database `gate.sh`, so nothing relies on jobs it brings in (lesson 13; orchestrator's decision). The land-alone list, the job-pattern sentence, the landing count (37) and the schedule updated; the span stays about 22.5 h. Claude P3-1: the type map's vector is the one definition of a decimal's canonical text, which the Python loader and the Rust binders each implement and run. Nothing else touched | Round 18's findings |
| 19 | Claude P2-1: one staging rule, defined once in T2.1a's runner and used by `gate.sh` and CI, runs the core legs on a database exactly when the built core under test reports a provider for it, otherwise the reference side only, with three controls in T2.1a; T2.2 and T2.3 edit no checkpoint-taken file, and T2.2b and T2.3b only create the `cross-host` jobs and `rust-test` passes (orchestrator's decision). P3s: the after-landing judging class named once in the process rules and the handbook (P3-1); 83 hours of product work (P3-2); the Python provider kit counts as a core leg under the same rule (P3-3). Nothing else touched | Round 19's findings |
| 20 | Round 20's three P3s only: a fourth T2.1a control naming the ratchet's "baselined, and not observed" guard; the after-product bullet moved below the T2.0c line; the providers reported through a private, plain-data marker (the native module's list for python-core and the Python kit, the test kit's provider list for rust-host, `rusqlite` included), never public API | Round 20's P3s, before the closure check |
| 21 | The owner's later answers only: KO2-47 (Q-L5, "A then."), KO2-48 (the key-cast vector's time-out outcome), KO2-49 (the `zip` advisory exception) and KO2-50 (`d > 1.5`); T2.17 gains the exact decimal-constant comparison exit with its neighbours and retires its temporary entry; new package CI8 lands alone, first, so the order is CI8, T2.0e, T2.0f, T2.0a, T2.0c (38 landings); frozen `t132g` fixtures keep their reference by an explicit list in the ratchet, carried by T2.0f's rebase | The owner's answers of 2026-10-16 and 2026-10-17 |
| 22 | Closure round 22's P2: the owner's "100% addressed" made mechanical in T2.1a's temporary-approval rule (KO2-45 and KO2-50 named once in `register.py`; locked check 1 fails, naming each approved entry that still cites one; seeded fault: a temporary entry left after its package lands), with T2.17, T2.18 and T2.8c each exiting with their temporary entries removed; KO2-45 and KO2-50 point to it. P3s: two owner-row docs landings, about 47 in all; KO2-47 cited for L5 | Closure round 22's findings |
| 23 | Round 23's P3s only: T2.0c registers the `d > 1.5` difference with `decision_ref` KO2-50; T2.8c, T2.17 and T2.18 retire their temporary entries (marked rejected or pointed at a permanent decision row, IDs kept) rather than remove them; the how-it-works notes leave KO2-45 and KO2-50, the cross-reference now in T2.0c's row | Round 23's P3s |
