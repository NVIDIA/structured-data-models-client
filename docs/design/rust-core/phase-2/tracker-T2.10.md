# T2.10: PQL targets

Lane f's worktree, started on the phase head after T2.8c's landing (34639b61), rebased onto T2.7's (e1fac9ee) with the test author's five commits (`T2.10 tests:`, a1bdb1e9..19154a3e) cherry-picked unedited.
The design's row: `aggregation`, `open_window`, `condition`, `logical_target`, `target_filter`, `static_target`, each as the reference does, with label vectors from the reference and cases per rule (the test author's, on `t210-tests`).
Carried: the deferred filter-role column-name cases and vectors (`contract/deferred/t210-column-filter/`); a ranking with `candidates=` naming an odd column and a ranking over a DECIMAL key refused under `target_filter` (both from T2.8's reviews, waiting on T2.8's landing, below).

## What the reference does, measured

Before writing anything, every target form the language parses was run through the reference's public calls at 6298913c (a route probe, below), to learn which forms its validation lets through for the relational model.
- `aggregation`: `COUNT_DISTINCT`, `FIRST`, `LAST` and a `LIST_DISTINCT` of any column but a foreign key are refused INVALID_REQUEST by the query's validation before anything is read; so are they on the core, whose validation is the reference's port.
  A `LIST_DISTINCT` of a foreign key is let through as a temporal link prediction (the test author's `target.aggregation.list-distinct`), which the design leaves to Phase 4: the core keeps it refused by name under `aggregation` (P2E-19, owner-only).
- `open_window`: a target window starting at `-INF` is refused by the validation ("TARGET cannot have an aggregation in the past"), a forecast's too.
- `static_target`: a column of another table, a table's `*` and a constant target are refused by the validation.
- `condition`: answered for a column of the entity's row of any type (text, boolean, number, decimal, time) against a constant of its type, with `=`, `!=`, `<`, `<=`, `>`, `>=` (ordering on numbers and times only) and `IN`, and for an aggregation with a comparison or `IN`; `LIKE`, `STARTS WITH`, `CONTAINS` and comparing two columns are refused by the validation or the grammar.
- `logical_target`: `NOT`, `AND` and `OR` of those conditions are answered, static or windowed (not mixed: the validation refuses that), across tables and windows.
- `target_filter`: `WHERE` inside an aggregation, over the aggregated table's own columns, with any condition above and `NOT`, `AND`, `OR`, is answered.

## What changed

- **The labels** (`sdm-engine` `labels`): the executor's conditions in full.
  A condition's constant is now a number, a boolean, text or a time (`Constant`); `IN` (`Target::Member`), `NOT` and `AND`/`OR` (`Target::Not`, `Target::Both`) join `Target`.
  `truth` gives each row whether a condition holds and whether it has every value the condition reads, as `execute_condition` and `execute_logical_operation` give it without dropping rows; a logical target labels only the rows every condition has a value for (`mask & right_mask`), and a `= NULL` inside one keeps only the rows with a value, as there.
  A condition that is the whole target reads an integer column at numpy's int64, so a float with a fraction in its `IN` list is the caller's error (`INVALID_REQUEST`, "Trying to coerce float values to integers"); inside a filter or a logical target it is compared exactly and matches no integer, a whole float matching its integer and one past int64 none (`int_member`, KO2-73, T2.10 fix 1).
  `aggregate_kept` aggregates the rows a filter keeps (`execute_filter`: the condition's values, a missing value failing a comparison, so `NOT` keeps it).
  Decimals are compared exactly in every condition and `IN` (KO2-35), times as nanoseconds.
- **materialize** (`sdm-core` `materialize`): the target is a tree (`Shape`: a column, an aggregation with an optional filter, a condition, `NOT`, `AND`/`OR`), built from the validated query; every form the validation refuses is now an internal error naming what got through (`unvalidated`), not a not-yet rule.
  Each aggregated table is read once (`fetches`, `_get_query_time_offset_dict`): by its one foreign key to the entity table, over the earliest start and latest end of its aggregations' windows, pooled as the reference pools a month against days (`pooled`, `_pooled_offset`), with every column its aggregations and filters read; each aggregation then applies its own window to those rows.
  The time bounds are those of every aggregated table; the context's anchor and the examples' step back are the target's merged window (`target_ast.date_offset_range`, the validated target's `range`).
  A date literal compared with a time column reads as its wall clock in nanoseconds (`time_constant`).
- **The rules lifted**: `open_window` (the forecast's too, `sdm-core` `pql::per_step`), `condition`, `logical_target`, `target_filter` and `static_target` leave the code, and `aggregation` keeps only the link prediction above; the five census probes leave `contract/refusal-census.json` and `aggregation`'s now probes the form that reaches it; the site check passes.
- **The refusal maps**: `empty-null.json` cites the line its refusal moved to.
- **Tests**: `labels`' own (logical masks, `IN` at each reading, a filter keeping a missing value under `NOT`, an aggregation in a logical target keeping each example's own value, times), `materialize`'s (the pools the reference gives, date literals), and `sdm-core`'s `tests/materialize.rs` no longer expects the three lifted rules.

## Probes

- **Route probe** (old route against core route, every public call, payloads compared byte for byte as parsed JSON), on DuckDB, with a test-ops wheel of this tree and the reference venv at 6298913c:
  - 137 hand-written queries over one source (`e`, `a`, `c`; every form above, each type, NULLs, decimals, times, `IN` quirks, windows of days, weeks, hours and months, two tables, two windows of one table in days and months): all agree in outcome, payload, predict and `_by_time` seeds (`differential.py probe` and `compare`), but two registered below (P2E-17, P2E-18) and one `parent_table` refusal (P1B-3, not this package's).
  - 50 generated sources (`differential.py generate 27 50`, each given a random target of the new forms; `differential.py probe` and `compare` on both routes, seeds included): 35 agree; 6 are P1B-11's empty tables; 7 differ in what `from_source` describes (a DECIMAL(38, 18) column's meaning) or a decimal feature's last digit, which this package does not touch (the same difference on a wheel of 34639b61 where that wheel answers, `g22`), not this package's; 3 are P2E-16 (two reference crashes, one misaligned label).
- **Cases on three databases**: 12 scratch cases over the `shop` base (filters, static and windowed logical targets, `IN`, a time filter, mixed units), `run_python.py` python-ref and python-core, then `compare.py`: on DuckDB, SQLite and Postgres no `sql.count` or `sql.result` difference (the per-table read is the reference's statement, with its rows) and no payload difference, but P2E-17's Postgres refusal.
  `compare.py`'s `inputs` rule flags the logical and filter cases because the Python host hands the core the query's canonical text (`NOT (users.age > 23)`) and the rule compares it with the case's `pql` as written; a case written in that form passes (a note for the test author).
- **The deferred filter role** (`contract/deferred/t210-column-filter/`): the three cases answer through `predict` on all three databases on python-core, and each of the twelve filter vectors answers as its twin `a_b` does on the reference (outcome, payloads, warnings, schema), the check of `test_t218_column_names.py`'s step 2.
- **Outside JSON**: no new reader of outside JSON.

## Deviations, registered (proposed, for the orchestrator)

- **P2E-19** (owner-only): `LIST_DISTINCT` of a foreign key without `RANK TOP`, a temporal link prediction the reference answers through materialize, stays refused by name until Phase 4.

- **P2E-16**: a logical target over an average, minimum or maximum some example has no value for: the reference packs those values onto the examples that have one and joins them by position, so later labels are other examples' (one generated source), or raises IndexError, INTERNAL_ERROR (two); the core joins each example's own conditions. Same class as KO2-70 #1.
- **P2E-17**: inside a filter or a logical target, an integer constant past a narrow integer column's type is refused by the reference on DuckDB (pyarrow cast) and a float with a fraction in an `IN` list of an integer column on Postgres (pandas Int64); the core, which reads no database's identity, compares exactly on every database: past-width integers as integers, and a float with a fraction matching no row where DuckDB and SQLite read it toward zero (fix 1, the owner's choice, KO2-73).
- **P2E-18**: a float with a fraction in an `IN` list of a whole-number decimal target: the reference reads the column as int64 and refuses; the core compares decimals exactly (KO2-35).
- The register IDs are from the top of lane E's block (P2E-16 to P2E-19), since T2.9 and T2.12 share the block; the orchestrator may renumber.

## Waiting on T2.8

T2.8 (lane d) lifts `target_filter` for a candidate's single comparison with a row filter of its own in `materialize.rs` (`RowFilter`, `filter_keeps`).
When it lands this package rebases over it: its filter becomes this one (`Windowed.filter`), which already reads decimals and any column name, closing the two carried ranking items (a candidate naming an odd column, a ranking over a DECIMAL key).

## Tests that held the old refusals

- `contract/t136-forecast-vectors.json`'s `count-filtered-3` (a forecast of a filtered count) holds python-core's old `target_filter` refusal; the core now answers it, as the reference does, so `contract/tests/test_t136_forecast.py` (two tests) and `sdm-testkit`'s `tests/t136_forecast.rs` fail until the test author moves the vector to the reference's answer (protected files, not the implementer's).

## The test author's tests

On a scratch merge of this lane with `t210-tests` (d5068a31, never pushed), with a test-ops wheel of this head:
- `contract/tests/test_t210_targets.py` on DuckDB, SQLite and Postgres: every key the pending list names passes now (112 tests fail only by naming themselves for removal from `t210-pending.json`), but `target.aggregation.list-distinct` (P2E-19), `rank.odd-column` (the Python host's ranking spells the candidate's column unquoted, refused INVALID_REQUEST: T2.8's, which moves the ranking into the core) and the three `name.*-filter` cases, which answer through predict as their twins do and wait only for P2E-1 (KO2-36) to name them, as it names T2.18's.
- `rank.decimal-key` passes on all three databases already: the client's candidate question filters on the DECIMAL key, which the filter reads exactly.
- `sdm-testkit`'s `tests/t210_targets.rs` and `tests/t218_column_names.rs` on DuckDB and SQLite: the same.

## After the rebase onto e1fac9ee

- **Goldens**: T2.4's run check over python-core's run of the target cases, the three name filter cases and the S0 boundary cases on DuckDB, SQLite and Postgres 17 found 21, 24 and 21 requests no golden held; each is now pinned in `sdm-data`'s `tests/<database>_goldens.rs`, whose render and run checks pass.
- **Register**: P2E-1 (KO2-36) names the three `name.*-filter` cases on both core runners and three databases, as it names T2.18's; P2B-16 (SQLite) and P2C-7 (Postgres), KO2-56's temporary entries for the four S0 boundary queries the not-yet rules refused until T2.10, are retired (rejected, IDs kept, each saying what replaced it).
- **Pending**: 141 keys leave `t210-pending.json` (`test_t210_targets.py`: 292 passed; `t210_targets.rs`: 3 passed); 21 stay: `rank.odd-column` and rust-host's `rank.decimal-key` (T2.8's ranking in the core), and `target.aggregation.list-distinct` (P2E-19).
- **A link prediction's answer**: the scripted endpoint answers none, so `sdm-request`'s unit test `a_link_prediction_s_answer_is_refused_by_name_never_read_as_another_task_s` holds the core's reading of one as the endpoint writes it (each item's `rankings`, a list of objects, which the reference's bindings read with `_mapping_tuple` and its result keeps in one row): against a link prediction's payload it is refused by name (`problem_type`), against another task's its `rankings` are refused by name (`response_field`).

## Gate against e1fac9ee

Both DuckDB legs (python-core with python-old, about 95 s; rust-host, about 93 s; each under the heavy lock) differ only on:
- `rank.decimal-key` and `rank.odd-column`, which wait for T2.8's ranking in the core;
- `target.aggregation.list-distinct`, refused by name under P2E-19 until the owner decides;
- `compare.py`'s `inputs` rule on python-core for `target.filter.and`, `target.logical.and` and `target.logical.not`: the Python host hands the core the query's canonical text (`PREDICT NOT (COUNT(orders.*, 0, 30, days) > 0) FOR EACH users.user_id`, `PREDICT (COUNT(orders.*, 0, 30, days) > 0) AND (SUM(orders.price, 0, 30, days) > 20) FOR EACH users.user_id`, `PREDICT COUNT(orders.* WHERE (orders.price > 18) AND (orders.price < 21), 0, 30, days) FOR EACH users.user_id`), the rule compares it with the case's `pql` as written; outcome, payload, statements and result are the reference's.
`prepush.sh` (cheap tier) passes.

## Rebased onto dd285821

Rebased onto T2.1o's landing (dd285821) with the test author's two fixes cherry-picked unedited (the three logical and filter cases' queries in the canonical text, and their python-ref lines recorded again).
Both DuckDB gate legs (about 90 s each, under the heavy lock) now differ only on `rank.decimal-key` and `rank.odd-column` (T2.8) and `target.aggregation.list-distinct` (P2E-19); the `inputs` differences are gone.
P2E-16 to P2E-19 stay proposed until the owner's rows (KO2-72 to KO2-75) land.

## Fix 1

The owner chose exact comparison for a float with a fraction in an `IN` list of an integer column inside a filter or a logical target (KO2-73): `int_member` gives no member for it, nor for a float past int64, and a whole float still matches its integer; a whole-target condition stays refused as the reference refuses it. `labels`' unit test expects `[T, T, F, T]` for `e.ni IN (1, 2.5) OR e.flag = true` and holds 2.0, 1e20 and -2.5 inside `NOT`. P2E-17's text says so. No case, label vector or baseline lists such a float against an integer column, so none moves.

## Rebased onto e154a8cc, the owner's rows in

- Rebased onto T2.1p's landing (e154a8cc), over T2.12 (contract v27; T2.10 adds no contract field, `contract_lock.py` clean). Conflicts resolved in the refusal census, the register, the empty-null map's citation, `tests/test_contract_cases.py` (the test author's file, T2.12's and T2.10's case lists side by side), the reference lines and the goldens; T2.12's NaN-key refusal now reads whether the target reads any table by key (`fetches`).
- P2E-16 to P2E-19 approved by the owner, KO2-72 to KO2-75 (each row names its entry); P2E-19 names `target.aggregation.list-distinct` on both core runners and three databases, with baselines on python-core and rust-host.
- Gate against e154a8cc (heavy lock, NVMe dirs): python-core and python-old 415 s, rust-host 138 s; both differ only on:
  - `target.aggregation.list-distinct`: the outcome is P2E-19's, but the core refuses at materialize where the reference stops at predict, so the comparator still holds the predict call's inputs and route, the payload, the warnings and the statement count. T2.1r (lane a) adds the opt-in for an earlier refusal; once it lands, P2E-19's rows name materialize under `statements_differ`.
  - `rank.decimal-key` and `rank.odd-column`: new cases of T2.10's test author, so they exist on no earlier head. They wait for the ranking in the core (T2.8): rust-host refuses every ranking (`problem_type`); python-core answers `rank.decimal-key` through the client's own decomposition, its question filtering on the DECIMAL key, and refuses `rank.odd-column`, whose question the client spells with the column unquoted.
- Client tests 2706 passed; mypy clean; `sdm-core`, `sdm-engine`, `sdm-request` tests 242 passed; goldens' checks pass; the register check passes. Pushed at 9bcf3694.
- Landing order (the orchestrator's): T2.8, T2.13, then T2.10 or T2.9, whichever is ready first; T2.10 rebases over T2.8 then (its `RowFilter` becomes `Windowed.filter`; the three forecast-vector files both packages change merged).

## Fix 2 (round 1's review)

- `s0.boundary.escapes@sqlite` and `@postgres` warn fewer times on the core route, which the gate's warnings rule shows now that P2B-16 and P2C-7 are retired. Confirmed through the public calls on SQLite: every extra warning is the same DeprecationWarning, "invalid escape sequence '\/'", from `predictive_query`'s `constant.py` line 134, raised each time the constant is read; the reference reads it five times (validation twice, its engine's labelling three times), the core route twice (validation only). Registered as P2E-15 (proposed, for the reviewers), not imitated (the orchestrator's decision). The comparator's warnings rule takes no register entry (only HOST-5's narrowing), so the case stays red on the warnings rule until a judging change takes P2E-15.
- What the gate differs on, per database (DuckDB from the local legs against e154a8cc; SQLite and Postgres from round 1's CI run at 9bcf3694 and the review):
  - DuckDB: `rank.decimal-key`, `rank.odd-column` (T2.8); `target.aggregation.list-distinct` (P2E-19's earlier refusal, T2.1r).
  - SQLite: the same three, and `s0.boundary.escapes@sqlite`'s warnings (P2E-15).
  - Postgres: the same three, `s0.boundary.escapes@postgres`'s warnings (P2E-15), and `cross-host (postgres)` stopping at `s0.boundary.depth-999@postgres`, whose 999-level filter the core now answers and so fetches past python-core's recording (T2.8's 3f8b86f1 refuses a diagnostic case past its recording; green after the rebase over T2.8).

## Rebased onto T2.8's landing (11d05883)

- T2.8's single-comparison row filter (`RowFilter`, `row_filter`, `filter_keeps`, `Values::retain`) is replaced by T2.10's general filter (`Windowed.filter`, `labels::truth`, `aggregate_kept`); T2.8's ranking questions take it unchanged. The three forecast-vector files are T2.8's (its re-record of `count-filtered-3` is the one T2.10's test author made), so the test author's 2378bfe8 became empty and was dropped. T2.8's ranking keeps `open_window` for a ranking window with no start (`ranking.rs`), which the validator refuses first; the census probes it.
- T2.8 fix 4 reads an answer's `rankings`: the link-prediction unit test now holds them kept against a task the core asks, and still refused by name (`problem_type`) against a link prediction's payload.
- `rank.decimal-key` passes on both core runners and three databases; `rank.odd-column` answers through predict as its twin does, named under P2E-1 (KO2-36, `items.i.d` and `buys.i.d` among its core-only columns). Their keys leave `t210-pending.json`; only LIST_DISTINCT's (P2E-19) stay. `test_t210_targets.py` 292 passed, `t210_targets.rs` 3 passed, clippy clean; baselines accepted for both ranking cases on the three runners.
- Postgres cross-host, run locally as CI runs it (python-core with `--record-fetches`, then `sdm-contract-run --replay`, `run_python.py legs`, `compare.py`): rust-host replays all 181 cases, `s0.boundary.depth-999@postgres` included; legs ok.
- What the gate differs on, per database, at this head:
  - DuckDB (both gate legs, through `bin/heavy`, against 11d05883): `target.aggregation.list-distinct` (P2E-19's earlier refusal: inputs, route, payload, statement count, warnings; T2.1r); `rank.odd-column` on python-core, see below.
  - SQLite (python-core and rust-host, legs ok): the same two, and `s0.boundary.escapes@sqlite`'s warnings (P2E-15; T2.1s).
  - Postgres (python-core and rust-host replay, legs ok): the same as SQLite.
- `rank.odd-column`'s two remaining differences are the comparator's reading of a core-only column, not the answer: the core's ranking calls carry `buys`'s `i.d` column, which the core-only-columns rule takes out of the request received but not out of `core.predicted.calls` (it reads a single call's body), so python-core's call and the request it sent differ by that column; and `from_source`'s key check (`SELECT k, COUNT(*), COUNT(DISTINCT key) ... UNION ALL ...`, statement 17) counts the `i.d` keys the reference leaves out, a result holding no column named `i.d`, so the rule's marking does not reach it. Both want a judging change.

## The key check numbers keys as the reference does (2026-10-20)

- The second `rank.odd-column` difference was the core's, not the comparator's: the reference leaves `items` out of its key check, since its key `i.d` holds a character its query language cannot say, and numbers the keys after it one lower, while the core gave `items` its place, so `orders` and `users` took `k` 2 and 3 on DuckDB (1 and 2 on SQLite) where the reference gives 1 and 2 (0 and 1).
- `count_keys` (`sdm-core/src/describe.rs`) now labels the keys through `key_labels`: a table whose key holds a column the reference leaves out (`` ` ``, line breaks, `.`, the reference's `_INVALID_NAME_CHARS`; no shared check for such names existed, so one small helper, `unsayable`, sits beside it) takes no place, and its key, still checked, takes a label past every one the reference gives. Unit test: `a_key_the_reference_leaves_out_takes_no_place_among_the_labels`.
- Checked: rust-host over all 183 cases on DuckDB against the previous run, only `rank.odd-column`'s fetches change (now `k` 1 and 2); on SQLite it now labels 0 and 1, as the reference. sdm-core tests and clippy clean.
- The per-call core-only-columns judging stays with T2.1s.
