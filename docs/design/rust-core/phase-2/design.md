# Phase 2: slice S1 offline

## Status

Draft for review (package T2.0a), on `rust-core-phase-2-design`.
The owner's kickoff answers go in [decisions.md](decisions.md); the acceptance checks below are locked when those answers are recorded and both reviewers approve the same SHA.
Everything not stated here is as in [Phase 1's design](../phase-1/design.md), cited as `P1 d:<line>`; Phase 1's decision rows are cited by ID and stay in force unless a row here or an owner answer changes them.
Owner questions are named `Q-G<n>` (process, phase-2-prep.md's G1 to G9), `Q2.<n>` (product), `K<n>` and `Q-<name>` (new here), so they never share a name with plan section 8's carried rows (G4, G6, G8 and so on).

## Goal

Ship slice S1 for Python: `Schema.from_source` → `materialize` → `predict` on DuckDB, SQLite and Postgres, offline, including ranking and forecasting, through the core (migration-plan.md section 4, S1).
Prove the data layer is swappable: a minimal non-SQL test provider passes the provider conformance kit, and the same prediction runs with an alternate data layer swapped in at the composition root, with no domain-crate edits (plan section 3, "Data access").
Exit: the slice gate (plan section 7, "Slice done"), checked by the locked acceptance checks below.
Paths stay unswitched for users: S0, S1 and S2 switch together in Phase 3 (plan section 4).

## Scope

1. **Backends.** SQLite and Postgres dialects in `sdm-data`, beside DuckDB; Python providers for both that hand Arrow to the core with no pandas round trip (plan section 3, "Performance rules"); rust-host providers for both (a dev-only `rusqlite` provider for SQLite; for Postgres, Q2.2).
2. **Operations.** Ranking (`top_k`, candidates, `prediction_statistic`, NaN ranks last under PD-1) with its decomposition moved into `sdm-request`; forecasting's remaining options (`lag_timesteps`, quantiles) on top of the split already in the core (E153); `main`'s multi-call `predict` (1fc5a686, below).
3. **The 37 named refusals of E92's class** (`phase-1/not-yet-refusals.md`; `contract/tests/test_t133_umbrellas.py` counts 37): each is lifted in Phase 2 or kept by an owner answer to Q2.4, which lists all 37. `float_nan` comes first (plan section 8, FR2 P3-1).
4. **The provider conformance kit** in `sdm-testkit` (fixture dataset, multi-page results, limits, expected answers, capability failure and retryability), passed by every built-in provider; the in-memory fallback for optional requests with declared limits that fail loudly; a minimal non-SQL test provider; a swapped `DataLayer`.
5. **The host-held semantics S1 reaches move into the core** (lesson 17). Each of Phase 1's ledger rows still marked for a later phase, with its disposition:
   - the ranking decomposition, its refusals and reassembly, and the forecast's `top_k` and `MOST_PARTS` refusals and parts' wording (`P1 d:263`): into `sdm-request` (T2.8a, T2.8b);
   - the answer re-encoding in `_encoded` and the response-depth coupling (`P1 d:277-278`): removed, the bytes cross untouched (T2.14b);
   - the model and route choice from the schema's shape (`P1 d:264`): stays in the host until tabular requests, Phase 4 (S3);
   - `main`'s multi-call `predict` (1fc5a686: batches, row numbering across calls, answered parts carried on a failed call, the 100,000-entity and 100-call ceilings, the batched-ranking and batched-forecast refusals): written in the core from the start (T2.8c), never in the host.
   The bridge freeze is widened to its full ledger first (FR P3-2) and lowered as the rows move.
6. **Reference `Request` pickles load** in the new client, and an unknown `RequestState` version is refused by name (H23; plan section 6, L2). `REQUEST_STATE_VERSION` and the version field exist (`sdm-contract/src/ops.rs:833-845`); what is left is loading the reference's pickles and versioning the state the new vocabulary adds.
7. **The full L4 and L5 for the three databases** (A5's deferral): SQL goldens executed on all three; the conversion tables on pandas 2 and 3; the same tables through three drivers reach the core as identical Arrow.
8. **Offline L7** (plan section 6): query counts equal to the reference's; CPU per scenario at most 1.1 times the reference's; the control-message size cap; zero-copy and "no pandas in providers"; import time and native library size within limits set from the first measurement. Live timing stays at the switch gate (Phase 3).
9. **The slice-gate chores**: the update from `main` and the reference rerun, at the start and again at the gate if `main` has moved; L3 on agreed seed sets; register reconciliation; the evidence pack; a fresh whole-phase review.
10. **Plan section 8's carried rows**, each assigned to a package or closed by a recorded decision (tracker, "Carried rows"), with T2.16 as the backstop for any row no lane reached (lesson 11).
11. **For information, not gating**: a dry run of Phase 3's flip in the phase's last hours (every route forced with `SDM_CORE_ROUTES`, the existing client suites run, failures counted), as later-phases-prep.md asks.

### Explicitly out

- Live warehouses, metric and semantic views, L6, live L7 timing and the Databricks job matrix: Phase 3 (`P1 d:41`).
- Frames and S3 reads, link prediction, sessions, tabular requests, the in-memory sampler, frame text inference (DT-6, PK-7): Phase 4 (`P1 d:41`, `P1 d:46`). E92's `model` and `text_time_rules` are proposed for Phase 4 too, in Q2.4, for the owner to confirm.
- Java: no slot in any phase as planned (question Java).
- The full L8 (`P1 d:43`); ASan and LSan move to Phase 5's L8 if the owner agrees (Q2.7).
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
   - `main`'s multi-call `predict`: several model calls per request in `RequestState`, the per-call and per-request limits, and an error's answered-parts fields;
   - the in-memory fallback's limits as declared capabilities; an undeclared optional request then takes the fallback instead of `UNSUPPORTED_REQUEST` (`P1 d:160`), and a declared capability that fails is still an error (`P1 d:161`);
   - `max_seed` required of every provider, with no "any seed" default (`P1 d:142`);
   - the `RequestState` version raised for the new state;
   - expression columns, unless Q2.7 defers them.
   A later need is a second bump renumbered by the lane that lands second (lesson 14; the renumbering script is T2.0b's, U13).
2. **The case format gets a backend axis** (T2.1a): one case names the backends it runs on, with accept and refuse vectors in `contract/tests/malformed-cases.json` run by both loaders (`P1 d:581`), plus the fields Phase 1's format could not carry (`candidates`, `top_k` on a forecast; E153).
   SQLite and Postgres bases are loaded from the DuckDB-built tables through Arrow (Q2.1), with a stated type map per engine: Postgres has no unsigned or 128-bit integers and SQLite no decimals, so each such DuckDB key or column lands as the reference's own load would put it, and the map is part of T2.1a's vectors.
3. **The reference capture extends to the reference's SQLite and Postgres drivers** (T2.1a), so the reference's statements and results on those backends are recorded like DuckDB's.
   Postgres runs locally in a pinned container (`POSTGRES_IMAGE`, already pinned in CI).
4. **Every new dialect renders every request kind**, including Phase 1's late ones (`column_ranges`, `repeated_key`, the typed seed join), and declares every capability with no fail-open default: `max_seed`, `null_order`, `index_advice`, `text_times`, `opaque_vendor` (`P1 d:142`).
   L0 already keeps dialect text inside `sdm-data` and the drivers off shipped crates (`P1 d:167`).
5. **Keys are typed per relationship** (lesson 19): a link joins in the joined key's declared type, both ends decided together; composite keys follow the same rule column by column.
   Every package that binds or casts keys runs the key-type pair matrix on its own engine before review: T2.2 (SQLite), T2.3 (Postgres), T2.12 (composite keys, all three).
6. **Ranking's decomposition and `main`'s multi-call `predict` live in `sdm-request`** (lesson 17), the way E153 moved the forecast split: the core splits a request into its questions and calls, refuses what the reference refuses, and puts the answer back together; the Python host keeps presentation only.
7. **The response bytes cross untouched**: the host stops re-encoding the answer (`_encoded`) and stops knowing the core's depth limit (`_ENCODED_DEPTH`), per plan section 3's "The host never parses or re-encodes the model request or response body".
8. **Sampling on SQLite and Postgres** follows the reference's registered streams: seeded SQLite draws use RS-2's stream (DF-2), and unseeded SQLite keeps `ORDER BY RANDOM()` (RS-2's notes; the `unseeded` rule); Postgres keeps the reference's `setseed` SQL under RS-6, which the reviewers approved at T1.36 fix 7 (Q2.5, Q2.6).
9. **The dataframe-engine comparison** (plan section 3, "Dependencies"; section 8's Phase 2 row): recommended waived (Q2.3), with the reason recorded: Phase 1 reached reference parity on plain Arrow kernels, and Phase 2's new data work (paging, the fallback's ordering and limits, the psycopg rows-to-Arrow path) is each a `phase-1/dependencies.md`-style row naming "use a library", "port a defined algorithm" or "keep in the host" with its measured reason (lesson 4), added by the package that writes it.
   If the owner wants the spike, it runs in lane A after T2.1b with its decision rule fixed before it starts.
10. **Refusals of inputs the reference accepts are generated, not surveyed** (lessons 8 and 18).
    T2.0d adds a census tool (`contract/tools/refusal_census.py`, new; `contract/runners/refusal_map.py` stays frozen at bug fixes, `P1 d:418`):
    - its rule list is generated from the code (every named refusal site in `rust/crates`), never written by hand, and it fails when a rule has no probe;
    - it runs each rule's probe on both routes (python-core against the reference) and on every backend the probe applies to;
    - it runs the public calls over the phase's real datasets (AdventureWorks and the RelBench sources T1.33 used) on both routes and counts, per rule, the real sources that hit it.
    It runs once before the owner's Q2.4 answer, so the answer can quote the counts, and again at every landing's gate and at T2.16. Locked check 2 fails on a refusal with no register entry and on a kept rule with a non-zero real-data count that no owner answer covers.
    It is new acceptance infrastructure, so the owner approves it (K5).
11. **Crash replacements cross every reading** (lesson 21): a refusal or check that stands in for a reference crash (E66(b)'s class) carries vectors across the UTC instant, the wall clock in the session zone, with and without optional libraries such as pytz, and each supported pandas, recorded from the reference and run on both core hosts, with a seeded fault that drops one reading.
    This is an exit of every package that adds such a refusal and a line of every test-author brief.

## Tasks

M is mechanical (ported, with reference vectors), N novel, R risky. "Judging" means the package changes a file or data `gate.sh` takes from the checkpoint (handbook, "Landing"); those land alone, first in their chain (lesson 13).
Each product package has a separate test author (briefs/test-author.md; lesson 16) who records the package's reference vectors, on the reference updated by T2.0c, before the implementer's first push.
Lanes: A head, B dialects, C data layer, D ranking and calls, E forecast and targets, F schema and options; the reference machine is the orchestrator's (see "Lanes and machine").

| ID | Package | Output | Exit check | Depends on | Lane | Judging |
|---|---|---|---|---|---|---|
| T2.0 | Tooling phase-aware; first checkpoint; rust-core's CI green | The phase a setting; every phase's decisions file read; the fork point as first checkpoint; an epic ratchet base; `--locked` from a per-phase locked-set file; the DuckDB extension install race fixed; watchdog hard failures; the Codex pane helper in the repo | Its self-tests and seeded faults; `rust-core`'s failing jobs green; a phase-2 lane lands | – | A | Yes |
| T2.0a | This design, decisions, tracker; learnings 17-22 | Locked checks; Phase 2's locked-set file | Two APPROVEs on one SHA, closure check, owner answers recorded | T2.0 | – | Yes (the locked-set file) |
| T2.0b | Carried tooling rows, and the reference-replacement rule | U4 (approvals by allowlist, not a `WITHDRAWN` denylist), U3, C2, C5, C22 (re-checked first), C28, G4, G10, FX-6 (the binding tests read the contract version), U13 (renumbering script), M5 (`TZ` in CI), FR P3-7's ratchet parts; the ratchet accepts a new reference commit only under an owner row in a phase decisions file naming it, and each changed reference line only with a register entry that cites the `main` commit it traces to (today it refuses any new reference commit, ratchet.py:1132-1138) | Each row's test fails before and passes after; seeded faults: a declined approval, a new reference commit with no owner row, a changed line citing no commit | T2.0 | A | Yes |
| T2.0c | Update from `main`, reference rerun | `rust-core-reference` merged with `main`'s 20 commits since fa0637d4 (no conflict; the four determinism fixes still apply), pushed by the owner (K2); the reference environment rebuilt and observations re-recorded; `main` merged into the phase branch (9 conflicting paths, three of them bridge-frozen: `client.py`, `materializers/relational.py`, `schema.py`, declared in `bridge_changes.json`); the core sends the inference options only when the caller set them, as `main`'s 26ecdd28 does (27 of the 52 S1 cases' requests change) | `gate.sh` against the new reference under T2.0b's rule; every changed line traced | T2.0b, K2 | ref | Yes (reference data) |
| T2.0d | Refusal census | `contract/tools/refusal_census.py` (choice 10) and its first run | Seeded faults: a refusal site with no probe, a probe skipped on one backend, a real-data run with a source left out each fail | T2.0 | A | No (evidence tool) |
| T2.1a | The S1 judging path | Case-format backend axis and the new fields with malformed vectors; the per-engine type map; reference capture on SQLite and Postgres; runners and comparator for three backends; comparator and both encoders holding a vector for every type in the type set and every stage a case can stop at (lesson 20); `coverage.py --gate S1` with G12, G16 and G17; L3 wiring (`differential.py` over three backends, M8, seed sets); C6, C18 | Seeded faults: a missing backend, a retyped field, an empty evidence file, an uncovered cell each fail; the post-landing gate passes judged against itself | T2.0c | A | Yes |
| T2.1b | Contract v26 | The vocabulary of choice 1, with M1, M2, M3 | Schema lock; both loaders; the binding jobs green on the bump (FX-6 landed) | T2.1a | A | No |
| T2.2 | SQLite | Dialect, Python provider (Arrow), rust-host `rusqlite` provider; U12; G6 | L4 goldens executed; the key-type pair matrix on SQLite; S1 cases on SQLite per Q2.1; the kit once T2.5 lands | T2.1b | B | No |
| T2.3 | Postgres | Dialect, psycopg provider (Arrow without pandas), rust-host provider (Q2.2); key sets bound as the dialect binds them (`P1 d:131`) | As T2.2, on Postgres | T2.1b | B | No |
| T2.4 | Full L4 and L5 | The goldens on all three databases; the Arrow-to-pandas presentation table on pandas 2 and 3 (never built in Phase 1, as-built T1.31); three-driver Arrow equality | CI L4 and L5 jobs; seeded type drift fails | T2.2, T2.3 | B | No |
| T2.5 | Conformance kit | Fixture dataset, paging, limits, expected answers (some hand-checked), capability failure, retryability; M9, G5 | DuckDB, SQLite and Postgres providers pass; a provider that truncates, drops a page or fails a capability silently fails the kit | T2.1b | C | No |
| T2.6 | In-memory fallback | Ordering, row limits, memory and transfer limits that fail loudly (Phase 4 reuses it) | Kit cases with the capability undeclared give the pushdown's answer; a limit exceeded is a named error | T2.1b | C | No |
| T2.7 | Non-SQL provider and swapped data layer | A minimal non-SQL test provider; a second `DataLayer` composed at the composition root | The provider passes the kit; the same prediction runs on the swapped layer; a guard shows no domain crate changed for it | T2.5 | C | No |
| T2.8a | Ranking, materialize side | `top_k`, candidates, the decomposition in `sdm-request`; `problem_type` for ranking; the forecast's `top_k` and `MOST_PARTS` refusals in the core | S1 ranking cases at materialize, both hosts | T2.1b | D | No |
| T2.8b | Ranking, predict side | Scores, rankings, `prediction_statistic`, PD-1, reassembly; `response_field` for rankings; `instance_id_type`, `prediction_type`; M10 | S1 ranking cases at predict, both hosts | T2.8a | D | No |
| T2.8c | Multi-call `predict` (`main`'s 1fc5a686) | Up to 10,000 entities a call, larger requests split, `rows_per_call`, rows numbered across calls, answered parts on a failed call, the 100,000-entity and 100-call ceilings, the batched-ranking and batched-forecast refusals | Cases: ordered multi-call results, a failure on a later call, each ceiling's boundary, both hosts | T2.8b | D | No |
| T2.9 | Forecasting | `num_forecasts`, `lag_timesteps`, quantiles; `response_field` for quantiles | Forecast cases on rust-host and python-core | T2.1b | E | No |
| T2.10 | PQL targets | `aggregation`, `open_window`, `condition`, `logical_target`, `target_filter`, `static_target` | Label vectors from the reference; cases per rule | T2.1b | E | No |
| T2.12 | Schema rules | `float_nan` first, `float_signed_zero`, `key_tie`, `key_name_order`, `composite_key`; FR P3-8's key-cast vectors; FX-3 if Q-FX3 keeps it in Phase 2 | The key-type pair matrix for composite keys on all three engines; cases per rule | T2.1b | F | No |
| T2.11 | Sampling options | `anchor_entity`, `context_anchor_time`, `exclude_cols_dict`, `num_hops`, `num_neighbors`, `max_pq_iterations`, `use_prediction_time`, `return_embeddings`, `unseeded`, `parent_table`; `response_field` for embeddings; M6; `main`'s eb6c68eb (a table reached by two paths, `sdm-core/src/subgraph.rs`) with new cases | Cases per option; `parent_table` on the RelBench sources that hit it | T2.12 | F | No |
| T2.13 | Query and option long tail | `entity_filter`, `entity_ids`, `assuming`, `evaluate`, `explain`, `target_transform_null`, `inference_config_text`, as Q2.4 decides; `response_field` for explanations and item metadata; U1 | Cases per rule | T2.1b | E | No |
| T2.14a | Bridge freeze widened | FR P3-2: `_send_body`, `_rust/codec.py`, `edits.py`, `errors.py` brought under the freeze, re-baselined | The freeze's self-test; its post-landing state judged against itself | T2.1a | A | Yes |
| T2.14b | Host clean-up | Response bytes untouched; `Request` pickles (H23); ledger rows deleted and the freeze lowered with `moved` declarations; the host wording rows G15, M4, FX-1, FX-4 if Q-W keeps them in Phase 2 | Pickle fixtures from the reference load; an unknown state version refused by name; `rust-host` runs every S1 case (lesson 17) | T2.14a, T2.8c | A | No |
| T2.15 | Offline L7 | A measurement tool over recorded runs (statement counts, CPU per scenario, message sizes, zero-copy and no-pandas checks, import time, native library size), its limits, and the first measurement | Seeded faults: an extra statement, a pandas round trip in a provider, a scenario slowed past 1.1 times | T2.1b | C | No (evidence tool) |
| T2.16 | The slice gate | Register reconciled; a second update from `main` if it moved since T2.0c; `gate.sh --locked 2`; L3; the census; offline L7; the held-out run; the evidence pack; the flip dry run; the whole-phase review; any carried row no lane reached | The locked checks below | all | A | No |
| HO2 | The new held-out set | About 40 seeded cases across the three backends, written by a held-out author (briefs/heldout-author.md) who reads no implementation, in T2.1a's case format, kept outside the repository and every worktree; a tested converter; its digest recorded; a dry run of `heldout_run.sh` on a dummy bundle | Frozen before any product package after T2.1b starts; only the orchestrator and the phase-end run read it | T2.1a, Q2.1, Q2.4 | orchestrator | No |
| CI5 | Faster CI | `rust-host` job split from `rust-test`; sharded `cargo test` and contract tests, each shard checking its list adds up to the whole; C4 | Same tests run (a seeded missing shard fails); timings recorded | T2.0b | A | Yes (`ci.yml`) |
| CI6 | Merge-queue landing | `land_chain.sh` reading a queue file, ejecting only the failing package and those above, `auto_land.py` running `mech_confirm.py` | Self-tests: an ejection rebuilds the stack; a clean move requeues | T2.0 | A | No (orchestration) |

The prep's T2.15 (coverage and L3 wiring) is folded into T2.1a, so the judging path is settled in one landing before any admission (lesson 20); its ID now names offline L7.
The test-author batch for T2.2 to T2.13 starts when T2.0c has landed and the owner's scope answers (Q2.1, Q2.4) are recorded.
The speed-ups that are not packages (one Codex pane per review step, a shared sccache, prebuilt wheels, warm non-editable venvs, `quick.sh`) are orchestration changes outside the tree; they change no check.

**Critical path**: T2.0 → T2.0b → T2.0c → T2.1a → T2.1b → T2.8a → T2.8b → T2.8c → T2.14b → T2.16.
Lane F (T2.12 then T2.11) and lane E (T2.9, T2.10, T2.13) each run three packages beside lane D's three, so they are not longer than it if their packages start at T2.1b's landing.

### Lanes and machine

The machine has 8 cores and 30 GB of memory, and a cold `cargo build --workspace` takes about 6 minutes; six lanes building at once would saturate it.
Run at most four product lanes at once (D always, as the critical path; then B, C, E and F as slots free), with a shared sccache, and keep each lane's Cargo target across landings (`SDM_CLEAN_AFTER_LAND` off, phase-2-prep.md item 3).

## Owner check-ins

- Kickoff: the questions below, answered at once; answers in decisions.md.
- Mid-phase: after T2.1b lands and the first product package of each lane is approved, with a design checkpoint (Q-G4) against learnings.md.
- End: the go or no-go packet (handbook, "Ending a phase", step 6).
- Twice a day, a five-line check-in; questions found mid-phase are marked owner-pending and work continues around them.

## The evidence pack

As Phase 1's (`P1 d:528-541`), on the final SHA, generated by the branch's own runs, plus:
- the per-case table by backend (DuckDB, SQLite, Postgres) and by host (python-core, rust-host);
- the refusal census (choice 10) with real-data hit counts;
- the conformance-kit results for every provider, the non-SQL provider and the swapped layer;
- the reference updates: the `main` commits merged and each changed reference line with the commit it traces to;
- L3: seed sets, counts and every divergence classified;
- offline L7 against its limits;
- the held-out run;
- the host-held ledger before and after (lesson 17);
- the flip dry run's counts, for information.

## Gates

### Merge rules

As Phase 1's (E25, E29, E43, E73(c), E130, E131, H2, H3), with the owner's answers to Q-G3, Q-G6, Q-G7 and Q-G8 applied once recorded.
Package IDs are `T2.<n>[a-z]` and `CI<n>` (land.py's pattern); HO2 is not a landing.
Register IDs and decision rows come from per-lane blocks: lane A register `P2A-1..19` and decision rows `A2-1..A2-19`, lane B `P2B-…` and `B2-…`, and so on, with kickoff answers `K2-…`; the IDs start with a capital letter, as `capture_check.py` and `coverage.py` read them.
Each lane writes only in its own register banner and decisions sub-table, and each package writes its own tracker file (question K4).

### Locked acceptance checks

Locked when the owner's answers are recorded (decisions.md); a third change goes to the owner (plan section 9).
Each is falsifiable: it passes on the candidate and fails on the seeded fault named, as Phase 1's were (`P1 d:555-564`; H26).
Phase 2's locked set is the per-phase file T2.0 makes `gate.sh --locked` read; T2.0a adds it, and its landing names that file in `Protected-paths`.

| # | Check | Tool and machine | Seeded fault that must fail it |
|---|---|---|---|
| 1 | Full S1 L2 parity: every S1 case in scope, on every backend Q2.1 names, per field, on python-core and rust-host against the reference; every difference has an approved register entry; `coverage.py --gate S1` with no empty cell | `gate.sh --locked 2` on the reference machine, plus CI's `contract-python` cells and `cross-host` | A changed SQL result, a dropped warning, a case missing on one backend, an empty coverage cell |
| 2 | No unapproved refusal of an input the reference accepts, and no panic; each of E92's 37 rules lifted or kept by an owner answer that quotes its real-data count | `ratchet.py` and the census (choice 10), reference machine | A rule re-added on one input; a refusal with no entry; a kept rule hit by a real source with no owner answer |
| 3 | L4: SQL goldens executed on DuckDB, SQLite and Postgres | CI | A seeded SQL change without a golden update |
| 4 | L5: conversion tables on pandas 2 and 3; three drivers give identical Arrow | CI | A seeded unit or scale change in one driver's path |
| 5 | Conformance kit: every built-in provider, the non-SQL provider and the swapped layer pass; the same prediction on the swapped layer with no domain-crate change | CI (`sdm-testkit`) | A provider that truncates a page; a capability that fails silently; a domain-crate edit in the swap |
| 6 | L3: `differential.py` on the agreed seed sets per backend against the reference, every divergence classified (lesson 12) | Reference machine | An unclassified divergence |
| 7 | The reference and the phase branch at `main`'s head as of the gate run, the reference rerun; each changed line traced to a `main` commit and registered | `ratchet.py`'s reference rule (T2.0b), reference machine | A new reference commit with no owner row; a changed line citing no commit |
| 8 | Host-held rows in scope moved: `rust-host` runs every S1 case; the freeze lowered; reference-pickled `Request` objects load and an unknown `RequestState` version is refused by name | `cross-host`, `bridge_freeze.py`, the pickle fixtures | A rule re-added on the Python side only; a corrupted pickle fixture; a bumped version |
| 9 | Held-out (HO2): the set under H11's threshold, on python-core and rust-host | `heldout_run.sh` once, at the phase end | Before the one shot, its dry run on a dummy bundle holding a panic and five unregistered differences must report a no-go |
| 10 | Boundary and packaging: L0 clean, no `pyo3` in the pure-Rust host's tree, the wheel installs and loads in a clean container | CI | Phase 1's seeded L0 faults (`selftest.py`) |
| 11 | Offline L7 within its limits (scope item 8) | T2.15's tool, reference machine | An extra statement; a pandas round trip in a provider; a scenario slowed past 1.1 times |

### Process rules that differ from Phase 1

- The design and docs land through the topic branch as T2.0a; every phase-branch commit is a landing (handbook, step 2).
- Judging-path packages (T2.0, T2.0b, T2.0c, T2.1a, T2.14a, CI5) land alone, before anything relies on them (lesson 13). T2.0c carries one core change (the inference options) with the re-recorded lines because each judges the other: the new reference lines fail the old core, and the changed core fails the old lines.
- The orchestrator runs the reference machine's work and HO2; implementers and test authors never open held-out data.
- E49(a) and E56(a) freeze tooling at bug fixes; T2.0 is a demonstrated integration failure (nothing lands on Phase 2, `rust-core` red); T2.0b, T2.0d, T2.15, CI5 and CI6 need the owner's answers (K1, K5).
- Whatever Q-G1, Q-G3, Q-G4 and Q-G6 to Q-G9 decide replaces the matching Phase 1 rows (E25, E29, E73(c)) for Phase 2 and later phases.

## Risks

| Risk | Guard | Lesson |
|---|---|---|
| The core grows pandas or driver behavior to match the reference on new backends | Providers hand Arrow to the core; differences go to the register; L5 checks three-driver equality | 1, 3 |
| Copying the reference's accidents (SQLite's unseeded order, NaN ranks) | Registered keeps and drops: RS-2, RS-6, PD-1; Q2.5 | 2 |
| Hand-rolled paging or fallback code | A `dependencies.md` row with a measured reason for each (choice 9); the kit judges it | 4 |
| Ranking, multi-call predict or the new backends meet the real host only at the gate | Both hosts run each case from its first landing; `cross-host` in CI; L3 before the gate | 5, 17 |
| Tests that check internals | Public-route cases by a separate test author; reference-recorded vectors | 6, 16 |
| A fix breaks neighbors (a dialect fix changes DuckDB) | Ratchet over every case on every backend | 7 |
| Hand-judged refusal lists | The census from the code (choice 10); coverage generated | 8, 18 |
| Process without brakes; lanes fragmenting | Stop-rule counters in the tracker; one lane per package; at most four product lanes | 9, 10 |
| Review agreements lost | Every finding ends as a fix, a carried row or a recorded "no"; T2.16 is the backstop | 11 |
| Differences found by review | Each implementer runs `differential.py` and the census and classifies every mismatch before review | 12 |
| Judging and product changes together | Judging packages land alone; T2.1a settles the judging path before admissions | 13, 20 |
| Contract bumps collide | One bump (T2.1b), holding `main`'s new vocabulary too; U13's renumbering script | 14 |
| Evidence on the wrong SHA | `land.py` exact-SHA rules; `final_gate.sh` | 15 |
| Key types across a relationship give wrong rows | Per-relationship typing; the key-type pair matrix in T2.2, T2.3 and T2.12 | 19 |
| A replacement for a reference crash misses a reading | Choice 11's crossed vectors, an exit of each such package | 21 |
| A benchmark on generated data | Q2.9 only on real data with a sanity floor; the live benchmark's `different` comparison fails the run (carried row G8) before any live rerun | 22 |
| Ranking needs two or more fix rounds | Split T2.8a, T2.8b, T2.8c; vectors in the test-author batch; result shape fixed in T2.1b | 7 |
| `composite_key` reaches key sets, the graph and the sampler | Its contract shape in T2.1b; if it slips, it stays refused by name and goes to the owner with its census count | 19 |
| The update from `main` changes behavior | T2.0c first; each new difference traced to a `main` commit; a second update at the gate | 2 |
| The machine saturates with parallel builds | At most four product lanes; shared sccache; CI5 | 10 |

## Owner questions

Each with a recommendation and its cost either way; the answers go in decisions.md as `K2-<n>` rows.
Process questions, once, for Phase 2 and later phases: Q-G1, Q-G3, Q-G4, Q-G6, Q-G7, Q-G8, Q-G9 (phase-2-prep.md's G1 to G9, with its recommendations), plus K1 to K5.
Product questions: Q2.1 to Q2.9 as phase-2-prep.md words them, corrected below, plus Q-FX3, Q-W, Q-M, Q-L7 and Java.

- **Q2.4** (rewritten): which of E92's 37 rules Phase 2 lifts. Recommended: all 35 of S1's, each in the package the task table names (T2.8a: `top_k`, `problem_type`; T2.8b: `prediction_statistic`, `response_field`, `instance_id_type`, `prediction_type`; T2.9: `lag_timesteps`; T2.10: six target rules; T2.11: ten sampling options; T2.12: five schema rules; T2.13: seven query and option rules), and `model` and `text_time_rules` kept until Phase 4, where frames and tabular requests are. If the owner keeps any of the 35, the answer quotes its census count (choice 10); cutting the long tail saves T2.13 (about half a day).
- **Q2.5** (corrected): the reference draws seeded SQLite samples with RS-2's stream (DF-2); only unseeded SQLite uses `ORDER BY RANDOM()`. The question is whether unseeded SQLite is compared on common inputs only. Recommended: yes; the alternative is a register entry for a draw nobody can repeat.
- **Q2.6** (corrected): RS-6 is already approved by the reviewers (T1.36 fix 7) for the Postgres route; the open point is only whether its approval covers the warehouse routes when Phase 3 ports them. Recommended: decide in Phase 3.
- **K1.** Tooling beyond bug fixes (E49(a), E56(a)): T2.0b, CI5 and CI6. Recommended: yes, each reviewed and landed alone like any judging change. A "no" leaves FX-6 and U13 (lesson 14's guard) to T2.1b and blocks T2.0c, whose reference commit the ratchet refuses without T2.0b's rule; CI stays at 43 to 88 minutes a push.
- **K2.** The reference update needs owner pushes: `rust-core-reference` moves to the merged `main`, recorded as an owner row naming the new reference commit, and the reference is re-recorded. Recommended: the orchestrator prepares the merge on a topic branch and the owner pushes it; re-record on this Linux machine, since every later comparison runs here, keeping the macOS recording's baseline tag. Not re-recording keeps cross-platform noise in every comparison.
- **K3.** A cache-warming workflow on `main` for topic-branch caches. Recommended: yes; it is the only change to `main`, and without it about 19 of 22 topic-branch runs build cold.
- **K4.** One tracker file per package, merged by a generator (phase-2-prep.md item 3, option A). Recommended: yes; it removes Phase 1's most common rebase conflict and changes no check.
- **K5.** The refusal census (T2.0d) and the offline L7 tool (T2.15) as new acceptance infrastructure (plan section 9). Recommended: yes; without the census, lesson 18 repeats, and without L7, the slice gate omits a layer the plan requires.
- **Q-L7.** Offline L7 in Phase 2 (scope item 8), or deferred to Phase 3's switch gate with live timing. Recommended: in Phase 2; it measures recorded runs only, and later-phases-prep.md asks for the CPU measurement before Phase 3.
- **Q-FX3.** Matching the reference for one value over 2 GiB (FX-3, E152) in Phase 2, or deferred. Recommended: deferred to Phase 4 (frames); no S1 dataset reaches it, and it costs a package.
- **Q-W.** The host wording rows (G15, M4, FX-1, FX-4) in Phase 2 (T2.14b), or deferred with E141's accepted wording. Recommended: in Phase 2; users see them, and each is a small host change.
- **Q-M.** What `main`'s update brings (T2.0c): keep the epic's `pyarrow>=14.0.1` floor (E16) where `main` leaves pyarrow unbounded, and take `main`'s other dependency bounds; take `main`'s new S1 behaviors (two-path sampling in T2.11, multi-call `predict` in T2.8c), matching the reference. Recommended: both. A decimal case whose reference answer flips from an internal error to an answer (eb6c68eb; P1B-18, P1B-31, P1B-33) is matched, under Q-G6(a) if the owner approves that class.
- **Java.** Its own track (about 12 to 15 packages), or deferred. Recommended: deferred until S2 is switched for Python; it doesn't block Python.
