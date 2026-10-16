# Phase 2: slice S1 offline

## Status

Draft for review (package T2.0a), on `rust-core-phase-2-design`.
The owner's kickoff answers go in [decisions.md](decisions.md); the acceptance checks below are locked when those answers are recorded and both reviewers approve the same SHA.
Everything not stated here is as in [Phase 1's design](../phase-1/design.md), cited as `P1 d:<line>`; Phase 1's decision rows are cited by ID and stay in force unless a row here or an owner answer changes them.

## Goal

Ship slice S1 for Python: `Schema.from_source` → `materialize` → `predict` on DuckDB, SQLite and Postgres, offline, including ranking and forecasting, through the core (migration-plan.md section 4, S1).
Prove the data layer is swappable: a minimal non-SQL test provider passes the provider conformance kit, and the same prediction runs with an alternate data layer swapped in at the composition root, with no domain-crate edits (plan section 3, "Data access").
Exit: the slice gate (plan section 7, "Slice done"), checked by the locked acceptance checks below.
Paths stay unswitched for users: S0, S1 and S2 switch together in Phase 3 (plan section 4).

## Scope

1. **Backends.** SQLite and Postgres dialects in `sdm-data`, beside DuckDB; Python providers for both that hand Arrow to the core with no pandas round trip (plan section 3, "Performance rules"); rust-host providers for both (a dev-only `rusqlite` provider for SQLite; for Postgres, Q2.2).
2. **Operations.** Ranking (`top_k`, candidates, `prediction_statistic`, NaN ranks last under PD-1) with its decomposition moved from the Python host into `sdm-request`; forecasting's remaining options (`lag_timesteps`, quantiles, `top_k` on a forecast), on top of the split already in the core (E153).
3. **The 37 named refusals of E92's class** (`phase-1/not-yet-refusals.md`, counted by `contract/tests/test_t133_umbrellas.py`; the prep's "35" leaves out `model` and `text_time_rules`, which are Phase 4's): each is lifted, or kept by an owner answer to Q2.4. `float_nan` comes first (plan section 8, FR2 P3-1).
4. **The provider conformance kit** in `sdm-testkit` (fixture dataset, multi-page results, limits, expected answers, capability failure and retryability), passed by every built-in provider; the in-memory fallback for optional requests with declared limits that fail loudly; a minimal non-SQL test provider; a swapped `DataLayer`.
5. **Host-held semantics in scope move into the core** (lesson 17): the ranking decomposition (`P1 d:263`), the answer re-encoding in `_encoded` and the response-depth coupling (`P1 d:277-278`). The bridge freeze is lowered to match and its coverage fixed (FR P3-2).
6. **Reference `Request` pickles load** in the new client, and `RequestState` carries a version (H23; plan section 6, L2).
7. **The full L4 and L5 for the three databases** (A5's deferral): SQL goldens executed on all three; the conversion tables on pandas 2 and 3; the same tables through three drivers reach the core as identical Arrow.
8. **The slice-gate chores**: the update from `main` and the reference rerun, L3 on agreed seed sets, register reconciliation, the evidence pack, a fresh whole-phase review.
9. **Plan section 8's carried rows**, each assigned to a package in the task table or closed out by a recorded decision (the mapping is in the tracker).

### Explicitly out

- Live warehouses, metric and semantic views, L6 and the Databricks job matrix: Phase 3 (`P1 d:41`).
- Frames and S3 reads, link prediction, sessions, tabular requests, the in-memory sampler, frame text inference (DT-6, PK-7), `model` and `text_time_rules`: Phase 4 (`P1 d:41`, `P1 d:46`; later-phases-prep.md).
- Java: no slot in any phase as planned; asked at kickoff (Java question).
- The full L7 and L8 (`P1 d:43`); ASan and LSan move to Phase 5's L8 if the owner agrees (Q2.7).
- The Python switch of any path: Phase 3.
- Expression columns, if only metric views need them (Q2.7).
- P1B-86 and P1B-89 stay as they are (E152).

## Architecture choices new in this phase

Phase 1's mechanisms are reused as built: the contract and its lock (`P1 d:100-125`), the data port (`P1 d:136-167`), runners and cells (`P1 d:319-360`), comparator (`P1 d:362-378`), ratchet and checkpoint (`P1 d:380-396`, H2, E131(f)), register v2 (`P1 d:398-404`), coverage (`P1 d:406-411`), gate (`P1 d:436`), lanes and landing (E43, E130, E131), the bridge freeze (`P1 d:249-258`), the evidence pack (`P1 d:528-541`).
What changes:

1. **One contract bump, v26, in T2.1b** (lesson 14; H4). All of Phase 2's vocabulary goes in it:
   - paging: a `Fetch` result may carry a continuation, and the core asks again until the provider says it is done; nothing is truncated silently (plan section 3, "The provider contract"), replacing Phase 1's "drained whole" (`P1 d:145`);
   - multi-column keys in `Neighbors`, `Lookup` and key sets (for `composite_key`), replacing the single `key: String` (`sdm-contract/src/data.rs:261-279`);
   - ranking and forecast result fields;
   - the in-memory fallback's limits as declared capabilities; an undeclared optional request then takes the fallback instead of `UNSUPPORTED_REQUEST` (`P1 d:160`), and a declared capability that fails is still an error (`P1 d:161`);
   - `RequestState` versioning (`REQUEST_STATE_VERSION`, `sdm-contract/src/ops.rs:833`);
   - expression columns, unless Q2.7 defers them.
   A later need is a second bump renumbered by the lane that lands second (lesson 14; the renumbering script is T2.0b's, U13).
2. **The case format gets a backend axis** (T2.1a): one case names the backends it runs on, with accept and refuse vectors in `contract/tests/malformed-cases.json` run by both loaders (`P1 d:581`).
   Bases for SQLite and Postgres are loaded from the DuckDB-built tables through Arrow (Q2.1), so one base definition feeds three engines.
3. **The reference capture extends to the reference's SQLite and Postgres drivers** (T2.1a), so the reference's statements and results on those backends are recorded like DuckDB's.
   Postgres runs locally in a pinned container (`POSTGRES_IMAGE`, already pinned in CI).
4. **Every new dialect renders every request kind**, including Phase 1's late ones (`column_ranges`, `repeated_key`, the typed seed join), and declares every capability with no fail-open default: `max_seed` (its range), `null_order`, `index_advice`, `text_times`, `opaque_vendor` (as-built digest; `P1 d:142`).
   L0 already keeps dialect text inside `sdm-data` and the drivers off shipped crates (`P1 d:167`).
5. **Keys are typed per relationship** (lesson 19): a link joins in the joined key's declared type, both ends decided together; composite keys follow the same rule column by column.
6. **Ranking's decomposition moves into `sdm-request`** (lesson 17), the same way E153 moved the forecast split: the core splits a ranking into its questions, refuses what the reference refuses, and puts the answer back together; the Python host keeps presentation only.
7. **The response bytes cross untouched**: the host stops re-encoding the answer (`_encoded`) and stops knowing the core's depth limit (`_ENCODED_DEPTH`), per plan section 3's "The host never parses or re-encodes the model request or response body".
8. **Sampling on SQLite and Postgres** follows the reference's registered streams: seeded SQLite draws as RS-2 (DF-2); unseeded SQLite keeps `ORDER BY RANDOM()` (the `unseeded` rule, RS-3 and RS-4); Postgres keeps the reference's `setseed` SQL under RS-6, which the reviewers approved at T1.36 fix 7 (Q2.5, Q2.6).
9. **The dataframe-engine comparison** (plan section 3, "Dependencies"; section 8's Phase 2 row): Phase 1 built on plain Arrow kernels with correct answers; recommended waived with that reason recorded (Q2.3). If the owner wants the spike, it runs in T2.1b's lane with its decision rule fixed before it starts.
10. **Refusals of inputs the reference accepts are generated, not surveyed** (lesson 18, lesson 8): `contract/tools/refusal_map.py` and `differential.py` produce, on both routes and on the phase's real datasets (AdventureWorks and the RelBench tables T1.33 used), the list of every refusal the core gives where the reference answers, with how often real data hits it. The evidence pack carries that generated list; a hand-written survey never counts.

## Tasks

M is mechanical (ported, with reference vectors), N novel, R risky. "Judging" means the package changes a file `gate.sh` takes from the checkpoint (handbook, "Landing"); those land alone, first in their chain (lesson 13).
Each product package has a separate test author (briefs/test-author.md; lesson 16) who records the package's reference vectors before the implementer's first push.
Lanes: A head, B dialects, C data layer, D ranking, E forecast and targets, F options and schema; the reference machine is the orchestrator's (see "Lanes and machine").

| ID | Package | Output | Exit check | Depends on | Lane | Judging |
|---|---|---|---|---|---|---|
| T2.0 | Tooling phase-aware; first checkpoint; rust-core's CI green | The phase a setting; every phase's decisions file read; the fork point as first checkpoint; an epic ratchet base; `--locked` from a per-phase locked-set file; the DuckDB extension install race fixed; watchdog hard failures; the Codex pane helper in the repo | Its self-tests and seeded faults; `rust-core`'s failing jobs green; a phase-2 lane lands | – | A | Yes |
| T2.0a | This design, decisions, tracker; learnings 17-22 | Locked checks; Phase 2's locked-set file | Two APPROVEs on one SHA, closure check, owner answers recorded | T2.0 | – | No (docs) |
| T2.0b | Carried tooling rows | U4 (approvals by allowlist, not a `WITHDRAWN` denylist), U3, C2, C5, C22 (re-checked first), C28, G4, G10, FX-6 (the binding tests read the contract version), U13 (renumbering script), M5 (`TZ` in CI), FR P3-7's ratchet parts | Each row's test fails before and passes after; seeded faults for U4 and FR P3-7 | T2.0 | A | Yes |
| T2.0c | Update from `main`, reference rerun | `main`'s 20 commits since fa0637d4 merged into `rust-core-reference` (no conflict; the four determinism fixes still apply) and into the phase branch (9 conflicting paths); the reference environment rebuilt and its observations re-recorded; the core sends the inference options only when the caller set them, as `main`'s 26ecdd28 does (27 of the 52 S1 cases' requests change), in the same landing as the re-recorded lines; every other new difference traced to a `main` commit and registered | `gate.sh` on the new reference; each changed line cites a `main` commit | T2.0 | ref | Yes (reference data) |
| T2.1a | The S1 judging path | Case-format backend axis with malformed vectors; reference capture on SQLite and Postgres; runner and comparator support for three backends; comparator and both encoders holding a vector for every type in the type set and every stage a case can stop at (lesson 20); `coverage.py --gate S1` with G12, G16 and G17; L3 wiring (`differential.py`, M8, seed sets); C6, C18 | Seeded faults: a missing backend, a retyped field, an empty evidence file, an uncovered cell each fail; the post-landing gate passes judged against itself | T2.0b | A | Yes |
| T2.1b | Contract v26 | The vocabulary above, with M1, M2, M3; `messages.py` derived (FX-6 landed) | Schema lock; both loaders; binding jobs green on the bump | T2.1a | A | No |
| T2.2 | SQLite | Dialect, Python provider (Arrow), rust-host `rusqlite` provider; U12; G6 | L4 goldens executed; the kit (once T2.5 lands); S1 cases on SQLite per Q2.1 | T2.1b | B | No |
| T2.3 | Postgres | Dialect, psycopg provider (Arrow without pandas), rust-host provider (Q2.2); key sets bound as the dialect binds them | As T2.2, on Postgres | T2.1b | B | No |
| T2.4 | Full L4 and L5 | The goldens on all three databases; the Arrow-to-pandas presentation table on pandas 2 and 3 (never built in Phase 1, as-built T1.31); three-driver Arrow equality | CI L4 and L5 jobs; seeded type drift fails | T2.2, T2.3 | B | No |
| T2.5 | Conformance kit | Fixture dataset, paging, limits, expected answers (some hand-checked), capability failure, retryability; M9, G5 | DuckDB, SQLite and Postgres providers pass; a provider that truncates, drops a page or fails a capability silently fails the kit | T2.1b | C | No |
| T2.6 | In-memory fallback | Ordering, row limits, memory and transfer limits that fail loudly (Phase 4 reuses it) | Kit cases with the capability undeclared give the pushdown's answer; a limit exceeded is a named error | T2.1b | C | No |
| T2.7 | Non-SQL provider and swapped data layer | A minimal non-SQL test provider; a second `DataLayer` composed at the composition root | The provider passes the kit; the same prediction runs on the swapped layer; a guard shows no domain crate changed for it | T2.5 | C | No |
| T2.8a | Ranking, materialize side | `top_k`, candidates, the decomposition in `sdm-request`; `problem_type` for ranking | S1 ranking cases at materialize, both hosts | T2.1b | D | No |
| T2.8b | Ranking, predict side | Scores, rankings, `prediction_statistic`, PD-1, reassembly; `response_field` for rankings; `instance_id_type`, `prediction_type`; M10 | S1 ranking cases at predict, both hosts | T2.8a | D | No |
| T2.9 | Forecasting | `num_forecasts`, `lag_timesteps`, quantiles, `top_k` on a forecast; `response_field` for quantiles | Forecast cases on rust-host and python-core | T2.1b | E | No |
| T2.10 | PQL targets | `aggregation`, `open_window`, `condition`, `logical_target`, `target_filter`, `static_target` | Label vectors from the reference; cases per rule | T2.1b | E | No |
| T2.11 | Sampling options | `anchor_entity`, `context_anchor_time`, `exclude_cols_dict`, `num_hops`, `num_neighbors`, `max_pq_iterations`, `use_prediction_time`, `return_embeddings`, `unseeded`, `parent_table`; M6; `main`'s eb6c68eb (a table reached by two paths, `sdm-core/src/subgraph.rs`) and 1fc5a686 (up to 10,000 entities a call, larger requests split, `rows_per_call`), each with new cases | Cases per option; `parent_table` on the RelBench sources that hit it | T2.1b | F | No |
| T2.12 | Schema rules | `float_nan` first, `float_signed_zero`, `key_tie`, `key_name_order`, `composite_key`; FR P3-8's key-cast vectors; FX-3 if Q-FX3 keeps it in Phase 2 | Key-type pair matrix on all three engines (lesson 19); cases per rule | T2.1b | F | No |
| T2.13 | Query and option long tail | `entity_filter`, `entity_ids`, `assuming`, `evaluate`, `explain`, `target_transform_null`, `inference_config_text`, as Q2.4 decides; U1 | Cases per rule | T2.1b | F | No |
| T2.14 | Host clean-up | Response bytes untouched; `Request` pickles (H23); ledger rows deleted; the freeze widened (FR P3-2) then lowered; the host wording rows G15, M4, FX-1, FX-4 if the owner keeps them in Phase 2 | Pickle fixtures from the reference load; the freeze's self-test; `rust-host` runs every S1 case (lesson 17) | T2.8b | A | Yes (`bridge_freeze.py` baseline) |
| T2.16 | The slice gate | Register reconciled; `gate.sh --locked 2`; L3; the held-out run; the evidence pack; the whole-phase review | The locked checks below | all | A | No |
| CI5 | Faster CI | `rust-host` job split from `rust-test`; sharded `cargo test` and contract tests, each shard checking its list adds up to the whole | Same tests run (a seeded missing shard fails); timings recorded | T2.0b | A | Yes (`ci.yml`) |
| CI6 | Merge-queue landing | `land_chain.sh` reading a queue file, ejecting only the failing package and those above, `auto_land.py` running `mech_confirm.py` | Self-tests: an ejection rebuilds the stack; a clean move requeues | T2.0 | A | No (orchestration) |

T2.15 of the prep is folded into T2.1a, so the judging path is settled in one landing before any admission (lesson 20).
The test-author batch for T2.2 to T2.13 starts when the owner's scope answers (Q2.1, Q2.4) are recorded; extending the reference capture (T2.1a) lands before it.
The speed-ups that are not packages (one Codex pane per review step, a shared sccache, prebuilt wheels, warm non-editable venvs, `quick.sh`) are orchestration changes outside the tree; they change no check.

**Critical path**: T2.0 → T2.0b → T2.1a → T2.1b → T2.8a → T2.8b → T2.14 → T2.16.

### Lanes and machine

The machine has 8 cores and 30 GB of memory, and a cold `cargo build --workspace` takes about 6 minutes; six lanes building at once would saturate it.
Run at most four product lanes at once (B, C, D, and E or F alternating), with a shared sccache, and keep each lane's Cargo target across landings (`SDM_CLEAN_AFTER_LAND` off, phase-2-prep.md item 3).

## Owner check-ins

- Kickoff: the questions below, answered at once; answers in decisions.md.
- Mid-phase: after T2.1b lands and the first product package of each lane is approved, with a design checkpoint (G4) against learnings.md.
- End: the go or no-go packet (handbook, "Ending a phase", step 6).
- Twice a day, a five-line check-in; questions found mid-phase are marked owner-pending and work continues around them.

## The evidence pack

As Phase 1's (`P1 d:528-541`), on the final SHA, generated by the branch's own runs, plus:
- the per-case table by backend (DuckDB, SQLite, Postgres) and by host (python-core, rust-host);
- the generated list of refusals of inputs the reference accepts, with real-data hit counts (choice 10, lesson 18);
- the conformance-kit results for every provider, the non-SQL provider and the swapped layer;
- the reference update: the `main` commits merged and each changed reference line with the commit it traces to;
- L3: seed sets, counts and every divergence classified;
- the new held-out set's run (below);
- the host-held ledger before and after (lesson 17).

## Gates

### Merge rules

As Phase 1's (E25, E29, E43, E73(c), E130, E131, H2, H3), with the owner's answers to G3, G6, G7 and G8 applied once recorded.
Package IDs are `T2.<n>[a-z]` and `CI<n>` (land.py's pattern).
Register IDs and decision rows come from per-lane blocks: lane A register `P2A-1..19` and decision rows `2A1..2A19`, lane B `P2B-…` and `2B…`, and so on (kickoff answers `2K…`); each lane writes only in its own register banner and decisions sub-table, and each package writes its own tracker file merged by a generator (phase-2-prep.md item 3, option A).

### Locked acceptance checks

Locked when the owner's answers are recorded (decisions.md); a third change goes to the owner (plan section 9).
Each is falsifiable: it passes on the candidate and fails on the seeded fault named (`P1 d:479`, H26).
Phase 2's locked set is the per-phase file T2.0 makes `gate.sh --locked` read; T2.0a adds it, and its landing names that file in `Protected-paths`.

| # | Check | Tool and machine | Seeded fault that must fail it |
|---|---|---|---|
| 1 | Full S1 L2 parity: every S1 case in scope, on every backend Q2.1 names, per field, on python-core and rust-host against the reference; every difference has an approved register entry | `gate.sh --locked 2` on the reference machine, plus CI's `contract-python` cells and `cross-host` | A changed SQL result, a dropped warning, a case missing on one backend |
| 2 | No unapproved refusal of an input the reference accepts, and no panic; each of E92's 37 rules lifted or kept by an owner answer | `ratchet.py` and the generated refusal list (choice 10) | A rule re-added on one input; a refusal with no entry |
| 3 | L4: SQL goldens executed on DuckDB, SQLite and Postgres | CI | A seeded SQL change without a golden update |
| 4 | L5: conversion tables on pandas 2 and 3; three drivers give identical Arrow | CI | A seeded unit or scale change in one driver's path |
| 5 | Conformance kit: every built-in provider, the non-SQL provider and the swapped layer pass; the same prediction on the swapped layer with no domain-crate change | CI (`sdm-testkit`) | A provider that truncates a page; a capability that fails silently; a domain-crate edit in the swap |
| 6 | L3: `differential.py` on the agreed seed sets per backend against the reference, every divergence classified (lesson 12) | Reference machine | An unclassified divergence |
| 7 | The reference updated from `main` and rerun; each changed line traced to a `main` commit and registered | Reference machine, `ratchet.py` reference check | A changed line citing no commit |
| 8 | Host-held rows in scope moved: `rust-host` runs every S1 case; the bridge freeze lowered with FR P3-2's coverage | `cross-host`, `bridge_freeze.py` | A rule re-added on the Python side only |
| 9 | Held-out: a new set of about 40 seeded cases across the three backends, unseen by implementers, under H11's threshold | `heldout_run.sh` once, at the phase end | Its dry run on a dummy bundle of the same format |
| 10 | Boundary and packaging: L0 clean, no `pyo3` in the pure-Rust host's tree, the wheel installs and loads in a clean container | CI | Phase 1's seeded L0 faults (`selftest.py`) |

### Process rules that differ from Phase 1

- The design and docs land through the topic branch as T2.0a; every phase-branch commit is a landing (handbook, step 2).
- Judging-path packages (T2.0, T2.0b, T2.0c, T2.1a, CI5, T2.14's baseline) land alone, before anything relies on them (lesson 13).
- The orchestrator runs the reference machine's work; implementers never open held-out data.
- E49(a) and E56(a) freeze tooling at bug fixes; T2.0 is a demonstrated integration failure (nothing lands on Phase 2, `rust-core` red); T2.0b, CI5 and CI6 need the owner's answer (question K1).
- Whatever G1, G3, G4, G6 to G9 decide replaces the matching Phase 1 rows (E25, E29, E73(c), H4) for Phase 2 and later phases.

## Risks

| Risk | Guard | Lesson |
|---|---|---|
| The core grows pandas or driver behavior to match the reference on new backends | Providers hand Arrow to the core; differences go to the register; L5 checks three-driver equality | 1, 3 |
| Copying the reference's accidents (SQLite's unseeded order, NaN ranks) | Registered keeps and drops: RS-2/3/4, RS-6, PD-1; Q2.5 | 2 |
| Hand-rolled paging or fallback code | Arrow kernels for the fallback; the kit judges it | 4 |
| Ranking or the new backends meet the real host only at the gate | Both hosts run each case from its first landing; `cross-host` in CI; L3 before the gate | 5, 17 |
| Tests that check internals | Public-route cases by a separate test author; reference-recorded vectors | 6, 16 |
| A fix breaks neighbors (a dialect fix changes DuckDB) | Ratchet over every case on every backend | 7 |
| Hand-judged refusal lists | Generated refusal list (choice 10); coverage generated | 8, 18 |
| Process without brakes; lanes fragmenting | Stop-rule counters in the tracker; one lane per package; at most four product lanes | 9, 10 |
| Review agreements lost | Every review finding ends as a fix, a carried row or a recorded "no" in the tracker | 11 |
| Differences found by review | Each implementer runs `differential.py` and classifies every mismatch before review | 12 |
| Judging and product changes together | Judging packages land alone; T2.1a settles the judging path before admissions | 13, 20 |
| Contract bumps collide | One bump (T2.1b); U13's renumbering script | 14 |
| Evidence on the wrong SHA | `land.py` exact-SHA rules; `final_gate.sh` | 15 |
| Key types across a relationship give wrong rows | Per-relationship key typing; the key-type pair matrix on three engines | 19 |
| A replacement for a reference crash misses a reading | Its vectors cross UTC, session zone, optional libraries and pandas versions | 21 |
| A benchmark on generated data | Q2.9 only on real data with a sanity floor; G8 fixed before any live rerun | 22 |
| Ranking needs two or more fix rounds | Split T2.8a and T2.8b; vectors in the test-author batch; result shape fixed in T2.1b | 7 |
| `composite_key` reaches key sets, the graph and the sampler | Its contract shape in T2.1b; if it slips, it stays refused by name and goes to the owner | 19 |
| The update from `main` changes behavior | Done first (T2.0c); each new difference traced to a `main` commit | 2 |
| The machine saturates with parallel builds | At most four product lanes; shared sccache; CI5 | 10 |

## Owner questions

Each with a recommendation and its cost either way; the answers go in decisions.md.
Process questions, once, for Phase 2 and later phases: G1, G3, G4, G6, G7, G8, G9 as phase-2-prep.md words them, plus K1 to K4 below.
Product questions: Q2.1 to Q2.9 as phase-2-prep.md words them, corrected below, plus Q-FX3, Q-W, Q-M and Java.

- **Q2.4** (corrected): the class is 37 rules, not 35; `instance_id_type` and `prediction_type` are in it and go to T2.8b.
- **Q2.5** (corrected): the reference draws seeded SQLite samples with RS-2's stream (DF-2); only unseeded SQLite uses `ORDER BY RANDOM()`. The question is whether unseeded SQLite is compared on common inputs only.
- **Q2.6** (corrected): RS-6 is already approved by the reviewers (T1.36 fix 7) for the Postgres route; the open point is only whether its approval covers the warehouse routes when Phase 3 ports them.
- **K1.** Tooling beyond bug fixes (E49(a), E56(a)): T2.0b, CI5 and CI6. Recommended: yes, each reviewed and landed alone like any judging change; without it, the carried tooling rows stay open and CI stays at 43 to 88 minutes a push.
- **K2.** The reference update needs owner pushes: `rust-core-reference` moves to the merged `main`, and the reference is re-recorded. Recommended: the orchestrator prepares the merge on a topic branch and the owner pushes it; re-record on this Linux machine, since every later comparison runs here, with the macOS recording kept as the baseline tag. Not re-recording keeps cross-platform noise in every comparison.
- **K3.** A cache-warming workflow on `main` for topic-branch caches. Recommended: yes; it is the only change to `main`, and without it about 19 of 22 topic-branch runs build cold.
- **K4.** One tracker file per package, merged by a generator (option A). Recommended: yes; it removes Phase 1's most common rebase conflict and changes no check.
- **Q-FX3.** Matching the reference for one value over 2 GiB (FX-3, E152) in Phase 2, or deferred. Recommended: deferred to Phase 4 (frames); no S1 dataset reaches it, and it costs a package.
- **Q-W.** The host wording rows (G15, M4, FX-1, FX-4) in Phase 2 (T2.14), or deferred with E141's accepted wording. Recommended: in Phase 2; users see them, and each is a small host change.
- **Q-M.** What `main`'s update brings (T2.0c): keep the epic's `pyarrow>=14.0.1` floor (E16) where `main` leaves pyarrow unbounded, and keep `main`'s other dependency bounds; take `main`'s two new S1 behaviors (two-path sampling, 10,000 entities a call with splitting) in T2.11, matching the reference. Recommended: both. A decimal case whose reference answer flips from an internal error to an answer (eb6c68eb; P1B-18, P1B-31, P1B-33) is matched, under G6(a) if the owner approves that class.
- **Java.** Its own track (about 12 to 15 packages), or deferred. Recommended: deferred until S2 is switched for Python; it doesn't block Python.
