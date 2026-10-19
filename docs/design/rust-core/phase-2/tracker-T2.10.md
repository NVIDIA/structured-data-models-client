# T2.10: PQL targets

Lane f's worktree, on the phase head after T2.8c's landing (34639b61).
The design's row: `aggregation`, `open_window`, `condition`, `logical_target`, `target_filter`, `static_target`, each as the reference does, with label vectors from the reference and cases per rule (the test author's, on `t210-tests`).
Carried: the deferred filter-role column-name cases and vectors (`contract/deferred/t210-column-filter/`); a ranking with `candidates=` naming an odd column and a ranking over a DECIMAL key refused under `target_filter` (both from T2.8's reviews, waiting on T2.8's landing, below).

## What the reference does, measured

Before writing anything, every target form the language parses was run through the reference's public calls at 6298913c (a route probe, below), to learn which forms its validation lets through for the relational model.
- `aggregation`: `COUNT_DISTINCT`, `FIRST`, `LAST` and `LIST_DISTINCT` are refused INVALID_REQUEST by the query's validation before anything is read; so are they on the core, whose validation is the reference's port.
- `open_window`: a target window starting at `-INF` is refused by the validation ("TARGET cannot have an aggregation in the past"), a forecast's too.
- `static_target`: a column of another table, a table's `*` and a constant target are refused by the validation.
- `condition`: answered for a column of the entity's row of any type (text, boolean, number, decimal, time) against a constant of its type, with `=`, `!=`, `<`, `<=`, `>`, `>=` (ordering on numbers and times only) and `IN`, and for an aggregation with a comparison or `IN`; `LIKE`, `STARTS WITH`, `CONTAINS` and comparing two columns are refused by the validation or the grammar.
- `logical_target`: `NOT`, `AND` and `OR` of those conditions are answered, static or windowed (not mixed: the validation refuses that), across tables and windows.
- `target_filter`: `WHERE` inside an aggregation, over the aggregated table's own columns, with any condition above and `NOT`, `AND`, `OR`, is answered.

## What changed

- **The labels** (`sdm-engine` `labels`): the executor's conditions in full.
  A condition's constant is now a number, a boolean, text or a time (`Constant`); `IN` (`Target::Member`), `NOT` and `AND`/`OR` (`Target::Not`, `Target::Both`) join `Target`.
  `truth` gives each row whether a condition holds and whether it has every value the condition reads, as `execute_condition` and `execute_logical_operation` give it without dropping rows; a logical target labels only the rows every condition has a value for (`mask & right_mask`), and a `= NULL` inside one keeps only the rows with a value, as there.
  A condition that is the whole target reads an integer column at numpy's int64, so a float with a fraction in its `IN` list is the caller's error (`INVALID_REQUEST`, "Trying to coerce float values to integers"); inside a filter or a logical target the source's type takes it toward zero, as on DuckDB and SQLite (`int_member`).
  `aggregate_kept` aggregates the rows a filter keeps (`execute_filter`: the condition's values, a missing value failing a comparison, so `NOT` keeps it).
  Decimals are compared exactly in every condition and `IN` (KO2-35), times as nanoseconds.
- **materialize** (`sdm-core` `materialize`): the target is a tree (`Shape`: a column, an aggregation with an optional filter, a condition, `NOT`, `AND`/`OR`), built from the validated query; every form the validation refuses is now an internal error naming what got through (`unvalidated`), not a not-yet rule.
  Each aggregated table is read once (`fetches`, `_get_query_time_offset_dict`): by its one foreign key to the entity table, over the earliest start and latest end of its aggregations' windows, pooled as the reference pools a month against days (`pooled`, `_pooled_offset`), with every column its aggregations and filters read; each aggregation then applies its own window to those rows.
  The time bounds are those of every aggregated table; the context's anchor and the examples' step back are the target's merged window (`target_ast.date_offset_range`, the validated target's `range`).
  A date literal compared with a time column reads as its wall clock in nanoseconds (`time_constant`).
- **The rules lifted**: `aggregation`, `open_window` (the forecast's too, `sdm-core` `pql::per_step`), `condition`, `logical_target`, `target_filter` and `static_target` leave the code; their six census probes leave `contract/refusal-census.json`, whose site check passes.
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

- **P2E-16**: a logical target over an average, minimum or maximum some example has no value for: the reference packs those values onto the examples that have one and joins them by position, so later labels are other examples' (one generated source), or raises IndexError, INTERNAL_ERROR (two); the core joins each example's own conditions. Same class as KO2-70 #1.
- **P2E-17**: inside a filter or a logical target, an integer constant past a narrow integer column's type is refused by the reference on DuckDB (pyarrow cast) and a float with a fraction in an `IN` list of an integer column on Postgres (pandas Int64); the core, which reads no database's identity, compares exactly and takes the float toward zero as DuckDB and SQLite do.
- **P2E-18**: a float with a fraction in an `IN` list of a whole-number decimal target: the reference reads the column as int64 and refuses; the core compares decimals exactly (KO2-35).
- The register IDs are from the top of lane E's block (P2E-16 to P2E-18), since T2.9 and T2.12 share the block; the orchestrator may renumber.

## Waiting on T2.8

T2.8 (lane d) lifts `target_filter` for a candidate's single comparison with a row filter of its own in `materialize.rs` (`RowFilter`, `filter_keeps`).
When it lands this package rebases over it: its filter becomes this one (`Windowed.filter`), which already reads decimals and any column name, closing the two carried ranking items (a candidate naming an odd column, a ranking over a DECIMAL key).
