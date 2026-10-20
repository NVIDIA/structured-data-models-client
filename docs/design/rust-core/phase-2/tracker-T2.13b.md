# Phase 2 tracker: T2.13b

The design branch owns `tracker.md`; this file holds T2.13b's section until it folds in.
On the test author's head 4b513615 (`T2.13b tests:` commits, branch `rust-core-phase-2-topic-t213b-tests`), which sits on T2.11's candidate d5afc6d5; this package's commits are `T2.13b:`.

## T2.13b

T2.13b lifts the core's two named refusals `entity_filter` and `assuming` (E92 rules, `UNSUPPORTED_REQUEST` at materialize), split out of T2.13 (`tracker-T2.13.md`, "Split").
Both conditions choose the examples by T2.10's evaluator; the core decides the client's notice and the reason a context is empty; the Python host only words them.

### What changed

- **The two refusals are lifted** (`sdm-core/src/materialize.rs`, `asked`).
  The entity filter's condition (`FOR EACH t.c WHERE ...`) and the assumption (`ASSUMING ...`) are read as a target's conditions are read (`condition_shape`, over the entity's own columns and its windowed aggregations, `IN`, `NOT`, `AND`, `OR`), and kept as `Asked.choosing`, the filter first.
- **They choose the examples as the reference's executor does** (`execute`: `mask & _mask`, the assumption's values and-ed into the filter's).
  Each condition is built over the examples from the rows read (`Labelling::build`) and evaluated by T2.10's `labels::truth`; its `values` (false where a value it tests is missing) are and-ed, then and-ed into the target's labels' mask, and the labels of the examples left out dropped (`Labels::kept`).
  No second evaluator: the target, the filter and the assumption are one `Shape`, one builder and one `truth`.
- **One read per table** (`Asked::every_windowed`, `fetches`): every table the target, the filter and the assumption aggregate is read once, over the windows of them all pooled, in the reference's order (`_get_query_time_offset_dict`: target, then entity, then assumption).
  The lags keep the target's own reads (`to_autoregressive_query` drops the filter and the assumption), and so do the anchor's time bounds (`get_all_target_aggregations`, the target's alone).
  The entity table's columns a condition tests are sampled with the entity rows (`all_query_columns`); a static target's exclusion stays the target's (`get_exclude_cols_dict`).
- **The notice that a filter chose only the examples** (`FILTER_CHOSE_EXAMPLES`), decided in the core: given on each materialize whose query filters its entity, after the query is validated, before anything is gathered, as the client's `_warn_the_filter_chose_only_the_examples` gives it.
  Its fields are the condition, as the reference writes it (`str(query.entity_ast.condition)`: `sdm-core/src/pql.rs`, `Validation.entity_filter`, by the parser's own `ast::to_string`), and the count of entities named.
  A ranking gives it once per candidate's question and a forecast once per step, as the reference's parts each do.
- **A context no example was collected for** (`empty_sample`): the reason the reference's sampler finds (`_diagnose_empty_sample`), as `reason` and the facts it names: `no_entity_at_anchor` (the entity table's earliest time after the context's anchor; `later` where an anchor after it leaves the target's window room), `no_entity_passed` (the data leaves room for the window after the anchor and an entity filter chose none: the condition, the anchor, the aggregated tables and their earliest and latest times), `anchor_too_late` (the data leaves no such room), else `too_restrictive`.
  The facts are the time bounds the run read (the memo), and the entity table's where it has a time column, read as the reference reads it.
- **Wording** (`_rust/errors.py`): `SAID['FILTER_CHOSE_EXAMPLES']` in the client's words, and `WORDED` for the four reasons in the sampler's words, chosen by `_worded` from `later` and the tables' count; a refusal's template is now filled by `_Words`, the formatter the warnings use, so a time is shown as the reference shows a `pd.Timestamp` and the tables as it lists them.
  Bridge changes declared (`errors.py::<module>`, `_worded`, `error_from_core`) and the freeze's baseline rewritten (`bridge_freeze.py --write`).
- **Elsewhere**: `contract/tests/t213-pending.json`'s `pending` is empty; `empty-null.json`'s citation of the empty-indices refusal moved with its line; the core's own refusal test now holds that both queries reach their first read.

### Exits

- Every key of the eight cases and the lag case passes on python-core (DuckDB, SQLite, Postgres) and rust-host (DuckDB, SQLite): `contract/tests/test_t213_options.py` 485 passed with a Postgres server named, `t213_options.rs` passed; the lag case (`lag.not-filtered-2`) passed already and keeps passing (`test_t29_forecasting.py`, `t29_forecasting.rs`).
- The refusal census: neither rule has a site left; the census's remaining failures are T2.11's ten lifted rules, fixed in T2.11's own test commit.
- Each refusal these two reach says what the reference says, word for word on all three databases: `filter.matches-none` (the anchor is not the reason, the filter named) and `assuming.no-examples` (too restrictive); the notice's words are the reference's for every filter case.
- L4: python-core's fetches for the nine cases on DuckDB, SQLite and Postgres are already goldens (`SDM_L4_OBSERVATIONS` over each run: `duckdb_goldens`, `sqlite_goldens`, `postgres_goldens` pass), so no golden is added.
- Baselines: the nine cases' lines on python-core, python-old and rust-host (27), by `ratchet.py accept` from the gate's DuckDB run (`gate.sh --base 4e5e26c9 --run python-core,python-old,rust-host`: observations, compare, l0 and coverage ok; the ratchet failed only on these missing baselines, and `ratchet.py check` passes once they are committed).

### Probes

- The reference (43e55f4f, semantically 6298913c), read before the code: `execute`'s mask, `_get_query_time_offset_dict`'s order, `to_autoregressive_query`, `get_exclude_cols_dict`, `_diagnose_empty_sample`, the client's notice, and `option_record.said`, which keeps each warning once (a ranking's three parts give one line).
- Differential, old route against core (`contract/tools/differential.py probe` and `compare`), on the 50 generated DuckDB sources of seed 27, each query given one of seven clauses: a numeric and a text entity column filter, a filter over an aggregation, an assumption, a negated `IN` filter with an assumption, a filter `OR` of a column and an `AVG`, and a filter no entity passes.
  - 33 agree; 6 differ as approved P1B-11 (an empty table, the reference's `RuntimeError`), the same without the clause.
  - 4 differ in the payload, every one the `OR ... AVG(...)` clause: the reference packs an average's values onto the examples that have one before it joins them to the other side (`execute_join` with `filter_na=True`), which misaligns them; the core aligns them, as approved register entry P2E-16 (KO2-72) records for a target's `NOT`, `AND` and `OR`. Rerun with `SUM` in place of `AVG`, all four agree.
  - 7 differ in the schema `from_source` infers, the same with the original generator spec and no clause (column `d`: categorical on the old route, numerical on the core): before any query, not this package's; reported to the orchestrator. Their materialize outcomes are the same on both routes.

### Deviations and questions

- None registered. The `AVG`, `MIN` or `MAX` alignment is P2E-16's difference of the shared evaluator, met here through an entity filter or an assumption, which P2E-16's `affected_inputs` names only for a target: whether to widen it to both clauses is the orchestrator's question (asked in the handoff), not entered here.
- The generated sources' stype difference above is pre-existing and outside this package.
