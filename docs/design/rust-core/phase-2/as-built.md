# Phase 2: as built

This file records where Phase 2 as built differs from [design.md](design.md) (handbook, "Ending a phase", step 5).
It reads the design's task table and choices, each package's `tracker-<ID>.md`, the owner's rows in [decisions.md](decisions.md) and the landing commits.
A package built as its design row says gets one line with its landing SHA.
Fix-round histories stay in the trackers; this file names only what changed against the design and where that change is decided.

Written at T2.16. The final tree is T2.16's own candidate: the phase head `f133c4b8` (`Land T2.14c`, after `Land T2.14b` 0b6d9a12) plus T2.16's docs, evidence, its one product fix (an answer's text encoding, below) and the `cells.sh` fix.
Phase 2 was merged into `rust-core` once mid-phase, through T2.1j, at `2e5fbd27`.

## Across the phase

**Coarser packages.**
Several design rows were built as one package by the orchestrator's decision: T2.5 with T2.6, T2.8a with T2.8b (as T2.8), and T2.2b, T2.3b and T2.4 (as T2.4).
T2.17 went inside T2.0c, and T2.15 inside T2.16a.

**Judging changes split out.**
Where a product package needed a judging change, the change landed first as its own package, T2.1d to T2.1v (the orchestrator's decision; lesson 13).
The design named none of them; they are listed under "Packages the design did not name".

**The contract.**
Choice 1 planned one bump, v26, holding all of Phase 2's vocabulary (lesson 14).
As built, v26 landed in T2.1b (`f6881b3d`) and then took optional fields from later packages: `SampleEntities.key_dtype` (T2.2), `IndexAdvice::LeadingColumn` and `DescribePart::IndexColumns` (T2.1b fix 8, for T2.2), and `Materialized.query` with `QueryFacts` (T2.18).
The mid-phase merge into `rust-core` (`2e5fbd27`) froze v26, so T2.12 opened v27 as the phase's working entry (`347ffa31`).
v27 holds `TableKey.label` (T2.12), the ranking fields and `RequestState.ranking` (T2.8), `lag_timesteps` in `Materialize.values` and quantiles as JSON object text in place of v26's list of float structs (T2.9), a nullable target transform (T2.13), `DescribedColumn.exact_number` (T2.21), and `Given::Spelled` and `TypedKind` (T2.14b).
`CONTRACT_VERSION` is 27 at this tree; `REQUEST_STATE_VERSION` is 2.
U13's renumbering script (T2.0b) was never built, since no two lanes opened a bump at once.

**`main` merged twice.**
T2.0c merged `main` at `327fda03` with the reference at `6298913c` (KO2-1).
T2.0i merged `main`'s two later commits, with the reference at `43e55f4f` (KO2-81, KO2-84); all 886 reference lines were equal under `ratchet.semantic`.

**Process rows that changed the plan.**
One design checkpoint mid-phase (MID2) replaced one every four or five packages (KO2-54); T2.21 and T2.14b's T2.14d came from it.
A fifth product lane ran (KO2-71).
Reviewers treat hardening of internal tools as P3 at most (KO2-52); tooling and docs fix rounds take one reviewer (KO2-57).
Owner-only product questions were decided overnight by the orchestrator under KO2-70, KO2-82 and KO2-88, each listed for the owner.
The S0 cases also run on SQLite and Postgres (T2.1a fix 1), so the case corpus on three databases is wider than the 52 locked S1 cases.

## The design's packages

### T2.0

Built as designed; landed `292f8d8a`, merged into `rust-core` as `3b81df85` (KO2-6), judged once by `bd44b342`'s harness as stand-in (KO2-5).

### T2.0e

Landed `912d9051`, as designed but for two choices recorded in [tracker-T2.0e.md](tracker-T2.0e.md).
An approval row must name exactly one full SHA, one step stricter than decisions.md's format.
The reference citations are an explicit table, `ratchet.CITATIONS`, after field discovery failed review twice (the stop rule).
U4's `WITHDRAWN` denylist was left to T2.0b, which was never built.

### T2.0f

Landed `b8f704e3`, as designed.
KO2-76 later reversed part of it: T2.1s (`ec64b077`) makes the reader refuse the steps `census`, `coverage-s1`, `l3`, `l7` and `ci` and the keys `limits` and `ci` by name.
Teaching `tests/test_contract_reference.py` the database keys was left to T2.1a, and fetching `origin/main` before the locked run to T2.16's runbook.

### T2.0a

Landed `5e972867` after 23 rounds of design review.
The locked set was changed once since, by T2.1s under KO2-76 (below).
Owner rows after it landed as docs landings of their own: T2.1d, T2.1k, T2.1p, T2.1s and T2.1u, and T2.14c for KO2-87 and KO2-88.
[tracker.md](tracker.md)'s package table was not updated after the first landings; T2.16 brought every row up to date and filled the stop-rule counters from the log.

### T2.0c

Landed `5a33dd7d`, from the topic branch `rust-core-phase-2-main-merge` rather than lane B ([tracker-T2.0c.md](tracker-T2.0c.md)).
It also carries T2.17 whole: the owner moved exact decimals into it on 2026-10-17 (KO2-35, KO2-50), registered as P2B-1, permanent.
`d > 1.5` is fixed here, so its temporary entry (P2B-6) and P2B-5 never landed.
Two gate fixes it needed landed first as T2.0g and T2.0h.

### T2.0d

Folded into T2.1a (`abd6846b`), which landed `contract/tools/refusal_census.py` and its DuckDB run.
It is a one-time report at the phase end, not a gate step (KO2-53, KO2-76).
The census records each probe against its intended site and claims no reach, after attribution from outcomes failed review twice (T2.1a fix 2).
It has no real-data mode; see "The locked checks as built".

### T2.0b

Never built: no tracker file and no landing.
Its gate-rule part became T2.0e.
Its carried rows (U3, U4, C2, C5, C22, C28, G4, G10, U13, M5, U14 and FR P3-7's ratchet parts) appear in no package's tracker; T2.16 is their backstop ([tracker.md](tracker.md), "Carried rows").

### T2.1a

Landed `abd6846b`, from lane f rather than lane A ([tracker-T2.1a.md](tracker-T2.1a.md)).
Moved out to a T2.1c by the owner's lean plan, and never built since KO2-53 and KO2-76 made them reports: `coverage.py --gate S1` (G12, G16, G17), L3 over three databases, the `coverage-s1`, `l3` and `ci` step tools, `--ci-run`, `gate.sh` running three databases, the common-input comparison (H24), C6 and C18.
python-old runs on DuckDB only; the SQLite and Postgres jobs hold python-ref's committed lines (T2.1a fix 4).
Those jobs ran in the reference cell only until T2.4.
The temporary-approval rule holds only a Phase 2 half run with `--wheel-run`, as `final_gate.sh` and `phase_end.sh` run it.

### T2.1b

Landed `f6881b3d`.
`TableKey.table` and `TableColumn.table` became relations too, against the brief, since the core renders each as a `FROM` ([tracker-T2.1b.md](tracker-T2.1b.md)).
v26 did not stay the phase's only vocabulary; see "The contract" above.

### T2.2

Landed `5efb35e2` ([tracker-T2.2.md](tracker-T2.2.md)).
The first wave's SQLite legs went to those packages, and the kit to T2.5 (the orchestrator's scope change).
A key set's temporary table is named by a 64-bit FNV-1a digest of its request, not SHA-256, so the shipped data layer takes no digest crate.
Its judging changes landed with T2.1a (fixes 7 and 8) and as T2.1e (`a921e455`).
Differences registered: P2B-10 to P2B-15, with the owner's KO2-59, KO2-61 and KO2-62.

### T2.2b

Folded into T2.4 (`8006efce`).

### T2.3

Landed `68f9b8e4` ([tracker-T2.3.md](tracker-T2.3.md)).
rust-host replays the Postgres server's answers to the core's own statements, recorded from python-core's run, not the reference's results (KO2-60, reading KO2-20).
The replay is judging, so it landed in T2.1d (`1a37efe2`).
The dialect issues the reference's own statements (fix 9), less the two unused reads (KO2-58).
H24 and its control went to the T2.1c that was never built.
Owner rows: KO2-58, KO2-59, KO2-60, KO2-63 and KO2-64.

### T2.3b

Folded into T2.4 (`8006efce`).

### T2.4

Landed `8006efce`, carrying T2.2b and T2.3b and running the SQLite and Postgres contract jobs in every cell ([tracker-T2.4.md](tracker-T2.4.md)).
No separate test author was assigned, by the orchestrator's decision.
`cross-host (postgres)` replays with `rust-test`'s build, handed on as an artifact.
Off DuckDB, python-core's host fields in the floor and dbr17 cells are reported, not gated, since python-ref's SQLite and Postgres lines exist only for the reference cell.

### T2.5

Landed `d339e88c` with T2.6 as one package ([tracker-T2.5.md](tracker-T2.5.md)).
The kit leaves out `time_bounds`, `column_ranges`, `sample_entities` and the catalog reads; the fallback's own tests hold the first three.
The kit's Python side runs on the test-ops build through `SDM_KIT_PYTHON`, since the shipped wheel has no single-request entry point.
On Postgres only the Python side runs the kit; rust-host's Postgres is a replay, not a provider.

### T2.6

Built inside T2.5 (`d339e88c`).
The fallback reads no text time, so a SQLite time column ends a fallback run by name.
At this tree `sdm-core/src/materialize.rs`'s `sampling_capabilities` still refuses a provider that does not declare `time_bounds`, `sample_entities`, `neighbors` and `lookup`, before any read, even one that declares the fallback (found at T2.7).
So the fallback answers those requests in the kit but cannot carry a `materialize`, which choice 1 says it should.

### T2.7

Landed `e1fac9ee` ([tracker-T2.7.md](tracker-T2.7.md)).
The swapped-layer prediction equals DuckDB's but for the drawn example rows, and equals the same provider on the default layer whole; the test author rewrote the test so (round 1).
The non-SQL provider declares `time_bounds` and `sample_entities` itself, because of the core check under T2.6.

### T2.8a and T2.8b

Built as one package, T2.8, landed `11d05883` ([tracker-T2.8.md](tracker-T2.8.md)).
It lifted `target_filter` for the one comparison a ranking question needs; T2.10 replaced that filter with its general one.
Its contract fields went to v27 at the rebase onto T2.12.
The batched-ranking refusal landed here, after T2.8c, as the design allowed.
`sdm_wire::response::mapping` became the one reader of mapping-valued response fields (the mid-phase check); T2.9 and T2.13 moved to it.
Register: P2D-1, approved by the owner (KO2-87).

### T2.8c

Landed `34639b61` ([tracker-T2.8c.md](tracker-T2.8c.md)).
Three judging parts landed first: T2.1g (`1486a8f1`, every call's body judged), T2.1i (`36bb2b73`, the case format) and T2.1l (`4404e28a`, faster validation of the 100,000-entity cases).
P2B-3 and P2B-4 are retired and P2B-9 rejected; P2G-1 records rust-host's `INVALID_RESPONSE` for an endpoint's refusal (E45).
Three rules keep the reference's class `StructuredDataError` (`AS_THE_BASE_CLASS`, P1B-20).

### T2.9

Landed `1d85bce3`, its case format first as T2.1m (`3cb8fd7f`) ([tracker-T2.9.md](tracker-T2.9.md)).
An average's, minimum's or maximum's lag sits on its own example where the reference shifts it (P2F-1, KO2-70).
P2B-18 and P2B-19 refuse lags past a time's range and non-finite lags by name.
The quantiles column became JSON object text in v27, since v26's shape could not keep a key as written.

### T2.10

Landed `fbdc6ee9` ([tracker-T2.10.md](tracker-T2.10.md)).
`aggregation` is lifted but for `LIST_DISTINCT` of a foreign key, a link prediction kept refused until Phase 4 (KO2-75, P2E-19).
Exact comparison where the reference misaligns, crashes or refuses: KO2-72, KO2-73 and KO2-74 (P2E-16 to P2E-18).
It also took T2.18's deferred filter role, and retired P2B-16 and P2C-7.
The key check now numbers keys as the reference does when a key holds a column the reference leaves out.
The owner rows landed first as T2.1p (`e154a8cc`).

### T2.12

Landed `347ffa31` ([tracker-T2.12.md](tracker-T2.12.md)).
`key_name_order` was removed as unreachable rather than lifted with a case.
A composite key is never sampled: a schema keyed by several columns is refused at materialize, without the reference's `folded_key` read (KO2-68, P2E-5); Postgres reads a declared composite key as no key (KO2-67, P2E-4).
The NaN and `-0.0` cases leave out SQLite, and two NaN cases Postgres (KO2-65).
It opened contract v27.
The interim edit gate's retirement, a later landing of its own in the design, never landed: `contract/tests/test_edit_cases.py`, `sdm-testkit`'s `tests/edit_cases.rs` and `contract/pinned/edits.core.json` are still in the tree.
Its owner rows and the `statements_differ` extension landed as T2.1k (`cfdbbaa4`).

### T2.11

Landed `c301dd1f`, kept in Phase 2 by the owner (KO2-79), its case format first as T2.1t (`6ae80743`) ([tracker-T2.11.md](tracker-T2.11.md)).
Choice 8 kept SQLite's unseeded `ORDER BY RANDOM()`; as built, an unseeded sample is drawn under a fresh seed each call, from the host's clock fact (P2I-2).
Option values are read as the reference's DuckDB SQL reads them on every database (P2I-1), and a negative hop size is refused by name (P2I-4).
All three were decided by the orchestrator under KO2-82.

### T2.13

Landed `4ad09482` ([tracker-T2.13.md](tracker-T2.13.md)).
`entity_filter` and `assuming` were split out to T2.13b, to build on T2.10's evaluator (the orchestrator's decision, 2026-10-19).
Recording a result's explanation in both runners was carried; T2.16a built it.

### T2.17

Folded into T2.0c (`5a33dd7d`) by the owner on 2026-10-17 (KO2-35, KO2-50); no landing of its own.

### T2.18

Landed `f42d92ea`, after its comparator rule T2.1f (`dfce343e`) ([tracker-T2.18.md](tracker-T2.18.md)).
The filter role moved to T2.10.
`Materialized.query` was added to v26 after T2.1b.
Register: P2E-1 (KO2-36); P2B-2 retired.

### T2.19

Built as designed; landed `dfdd532a`.
The `diamond` and `chain` shapes sampled from a table with a foreign key were left to T2.11's `parent_table`.

### T2.14a

Landed `24f5be66`, in two parts ([tracker-T2.14.md](tracker-T2.14.md), "T2.14a").
T2.1v (`4e5e26c9`) landed first, so the checkpoint's copy of the freeze judges only what it freezes; it first landed as "T2.14a1", an ID outside the package pattern, and was rolled back.
The freeze also covers `_rust/providers/postgres.py`, beyond the design's list, and gained a strict mode for a copy judging its own tree.

### T2.14b

Landed as 0b6d9a12.
T2.14d (the Postgres value rules, MID2 P2-3) and the decimal stype fix ride in it, by the owner's granularity decision.
Errors may name a value the caller passed in that call (KO2-86) and M4's data times (KO2-89).
The candidate-repeat rule stays the host's, by Python's own equality, sent as the contract's `repeat` (fix 7); `PredictiveQuery._resolve` stays in the ledger.
Four items are carried to Phase 3 (the tracker's "Carried to Phase 3").

### T2.14c

Landed as f133c4b8.
It lands with product items rather than alone, since T2.16a's fix 4 taught the checkpoint's freeze to accept a declared `gone` entry.
The freeze was lowered by measurement: 29 functions leave `COVERED`, five stay because `selftest.py` seeds faults in them.
It carries a Postgres numeric-key fix (P2G-4, P2G-5), `labels::compare`'s deletion, KO2-87 and KO2-88, P2D-2 to P2D-5, and T2.16a's part B (P2O-1).

### T2.15

Folded into T2.16a as `contract/tools/l7.py`, a report against KO2-29's limits with no seeded-fault suite (KO2-53, KO2-76).
It is not in `contract/protected-paths.txt`.

### T2.16

The final package: the phase-end docs (KO2-90 and KO2-92 for check 2's kept rules, KO2-91 for the lean plan, the tracker's true statuses, this file, the "Carried into Phase 3" rows), the evidence pack, and two fixes.
- **An answer's text encoding** (the whole-phase review part 2, a regression since T2.14b): since T2.14b the HTTP transport hands the core the answer's bytes, and `sdm_wire::json::read_answer` refused UTF-8 with a byte order mark, UTF-16 and UTF-32, which the reference's `json.loads` reads.
  `read_answer` now decodes the bytes first as `json.detect_encoding` does (`answer_text`), in the core, so a Java host reads alike.
  The Databricks transport encodes the served answer's text with `surrogatepass`, so a raw lone surrogate reaches the core and is refused by name.
  Register P1B-24 is widened to raw lone surrogates (the orchestrator's decision under KO2-88; on Databricks this refuses an answer the reference accepts).
- **`scripts/rust_core/cells.sh`** installs the connectors with the `sqlite` and `postgres` extras, as `ci.yml`'s "Install the cell" step does, with one test.
Its gaps found at the phase end are in the next section.

### HO2

Not a landing; written in parallel by a held-out author (KO2-55).
T2.16a built its converter and check 9's dummy bundles, and reads H11 per case across database legs (the orchestrator's reading, 2026-10-21).
The held-out record's stored-NaN and `key_name_order` items are not in the set.

### CI5

Dropped by the owner on 2026-10-17 (the lean plan); C4 went with it.

### CI6

Dropped by the owner on 2026-10-17 (the lean plan, KO2-91).
Landings stayed on `land.py`, which confirms a clean identical move without a review round (KO2-80).

### CI7

Landed `3d7aaf8f`, kept in Phase 2 by KO2-79 ([tracker-CI7.md](tracker-CI7.md)).
The DBR 16.4 cell starts with no pandas, since its runtime's pandas 1.5.3 has no installable build, and DBR 14.3's cell preinstalls no PyYAML.
The aarch64 build image came from quay.io by digest; T2.0i moved it to a ghcr.io copy.
Register: P2J-1 and P2J-2 (KO2-87).

### CI8

Built as designed; landed `8bcc85e4` (KO2-49).
Its watch item was not added to a dependencies file.

## Packages the design did not name

- **T2.0g** (`95259022`): the gate fixes T2.0c's merge needed, a covered function `main` deleted, `more_exact` fields and new-case moves (the owner's "A ok", 2026-10-17).
- **T2.0h** (`5520d634`): the floor cell named at T2.0c's new pins (KO2-37).
- **T2.0i** (`2df155b6`): `main`'s second update (KO2-81, KO2-84).
- **T2.1c**: planned at T2.1a for the reporting tools; never built (KO2-53, KO2-76).
- **T2.1d to T2.1v**: judging and owner-row landings, each alone: T2.1d `1a37efe2`, T2.1e `a921e455`, T2.1f `dfce343e`, T2.1g `1486a8f1`, T2.1h `54b0d828`, T2.1i `36bb2b73`, T2.1j `c8075715`, T2.1k `cfdbbaa4`, T2.1l `4404e28a`, T2.1m `3cb8fd7f`, T2.1n `cb8d6fb4`, T2.1o `dd285821`, T2.1p `e154a8cc`, T2.1q `b2b0f36c`, T2.1r `44e1c9a7`, T2.1s `ec64b077`, T2.1t `6ae80743`, T2.1u `fedaf7cd`, T2.1v `4e5e26c9`.
- **T2.13b** (`b2732c12`): the entity filter and ASSUMING, on T2.10's evaluator; P2E-16 widened to both clauses (KO2-85).
- **T2.14a1**: T2.1v's first landing under an invalid ID, rolled back and landed again as T2.1v.
- **T2.14d**: rides in T2.14b.
- **T2.16a** (`9a899950`): the final gate's tooling, held-out legs per database, the converter, explanations on both hosts, L7, coverage reading a target behind `EVALUATE` and `EXPLAIN`, and the kept-rules guard.
- **T2.21** (`3e2342f2`): exact numeric keys past 2**53 on Postgres, from MID2 (KO2-78, P2G-2; P2G-3 under KO2-35).

## The locked checks as built

KO2-76 (2026-10-20) changed the locked set: the `census`, `coverage-s1`, `l3`, `l7` and `ci` steps left the gate, since KO2-53 had made their tools end-of-phase reports.
T2.1s landed that change (`ec64b077`); it is the one change since T2.0a.
The half `2a` holds the 52 S1 cases, the steps `inventory`, `seeded` and `main`, and checks 1, 2, 7 and 10, with 12 as the `every` check.
Check 1's `coverage` step is the existing S0 coverage step.
The CI-green parts of checks 1, 3, 4, 5, 8 and 10 are judged by `land.py`'s exact-SHA rule.
The census, coverage S1, H24, L3 and L7 are reports read by both reviewers, and check 9 runs once through `phase_end.sh`.
H24 went from gating (KO2-25) to a report, and unseeded SQLite sampling is judged by properties (KO2-76).

Measured at T2.16, true at the final tree:

- `scripts/rust_core/coverage.py` has no `--gate S1`: rule 7 is not built and `ACTIVE` is still `('python', 'S0')`.
  So the S1 coverage report is generated from `coverage.py`'s own `generated()`.
- `contract/tools/differential.py` (L3) runs DuckDB sources only, so L3 on SQLite and Postgres is not measured.
- `scripts/rust_core/evidence.py` and `scripts/rust_core/orchestration/phase_end.sh` read a Phase 1b inventory (`cases: [{id: ...}]`).
  Phase 2's locked set lists `halves.2a.cases` as plain ids, and `evidence.py` has no per-database legs.
- `scripts/rust_core/cells.sh` installed the connectors with only the `duckdb` extra, while `ci.yml` installs `[duckdb,sqlite,postgres]`; T2.16 fixed it.
- The census tool has no real-data mode, so the kept rules' counts are zero by construction (T2.16a, `evidence/census/real-data.md`, KO2-90).
- No tool for H24's Postgres common-input comparison exists: nothing under `contract/tools`, `contract/runners` or `scripts/rust_core` names it.
- Three trackers (T2.8c, T2.10, T2.13b) reported seven old-against-core differences on the generated DuckDB sources in what `from_source` infers for one column (`d`, categorical against numerical). At be55838a the inferred schema agrees on every source; L3 there finds the payload's one-ulp decimal differences under P2B-1 instead (`evidence/l3/classification.md`). Which landing closed the stype difference was not bisected; T2.14b's decimal stype fix is the likely one.
- Each gap above has a "Carried into Phase 3" row in migration-plan.md section 8 (P2-T2.16-3 onward).
