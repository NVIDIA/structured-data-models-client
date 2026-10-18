# Phase 2 tracker: T2.7

The design branch owns `tracker.md`; this file holds T2.7's section until it folds in.
Lane c, on the phase head after T2.5's landing (d339e88c, T2.5 with T2.6), then the test author's acceptance tests (`T2.7 tests:`, cherry-picked from `t27-tests`), then this package (`T2.7:` commits).

## T2.7

T2.7 proves the data layer is swappable: a minimal non-SQL test provider that answers the core's typed data requests from in-memory Arrow tables, and a second `DataLayer`, for host providers alone, composed at the composition root in place of `sdm-data`'s.
Both are dev-only, in `sdm-testkit`; nothing shipped reaches them (`l0_deps.py`: nothing shipped reaches `sdm-testkit`).

### What changed

- **The non-SQL provider** (`sdm-testkit/src/memory.rs`, new): `Memory`, a list of `Table`s, each its name, its rows as one Arrow batch, and its catalog as plain data (columns as declared, keys, foreign keys).
  It answers every request from those batches with Arrow's own kernels (`arrow-select` to filter and take, `arrow-row` to group, join and order), never a statement.
  A relation names a table by one part, the name exactly; any other relation, or a name it does not hold, fails as the source's (`data_source`).
  It declares every optional request but `column_ranges`: the kit's six and `time_bounds` and `sample_entities`, which the core's sampler requires before it reads anything (below).
  Values compare as the SQL pushdowns compare them: a key's parts all present to match; integers of any width and decimals of scale 0 by value; `-0.0` and `0.0` one value, every NaN one; a time as `CAST(column AS TIMESTAMP)` reads it (a zone-less timestamp, a finer unit truncated to microseconds, or a date at midnight).
  What it does not read (another time type, seeds of another key type, a filter other than not-null) fails the request, `unsupported`, never answered otherwise.
  A sample keeps the table's order: every row where `n` covers them, else its own seeded draw (SplitMix64, Fisher-Yates), stable under a seed as it declares.
  `Memory::from_duckdb` copies a DuckDB database's tables and catalog into one, once, to build a fixture; the provider never runs a statement after.
- **The second data layer** (`sdm-testkit/src/layer.rs`, new): `HostLayer` implements `sdm_contract::data::DataLayer` apart from `sdm-data`: a host provider's declared capabilities, `FetchPlan::Host` with the contract's expected schema, and its own reader of a host provider's catalog answer; a SQL dialect is refused, `UNSUPPORTED_REQUEST` naming it.
  `layer::root()` composes it: `CompositionRoot::new(Arc::new(HostLayer))`.
- **The composition root's type** (`sdm-host/src/lib.rs`): `sdm-host` re-exports `sdm_core::CompositionRoot` beside the other types a binding hands across (E21), so a host composes its own data layer; `default_root` is unchanged.
  `sdm-host` is a host crate, not a domain crate.
- **The runners**: `sdm-contract-run --kit memory` runs the kit on the provider (a host provider on the default root, as any host provider runs); `sdm-contract-run --database memory` runs each case that reads data over its DuckDB base copied into the provider, on `layer::root()`, keyed `<case>@memory` (`host::run_memory`).
  The pure-Rust host's runs take their root from their source, and a host plan's key sets come from the request itself.
- `sdm-testkit` takes `arrow-array`, `arrow-row`, `arrow-schema` and `arrow-select` as normal dependencies (already in the lockfile; no first-party edge added).

### Exits

- **The provider passes the kit, including two schema tables over one source**: `t27_non_sql_provider.rs`'s `the_kit_passes_on_the_non_sql_provider` passes (every kit case, `key-counts.one-source-twice` among them), and `a_seeded_fault_fails_the_kit_on_the_non_sql_provider` passes (truncation and a silenced capability each fail the cases `expected.json` names).
  Beyond the kit, `memory_provider.rs`'s `two_schema_tables_over_one_source_predict_as_two_sources`: `shape.same-table-twice.buyers` with both schema tables reading `customers` (and `buyers` dropped) predicts exactly as over the two equal tables.
- **The same prediction runs on the swapped layer**: not met as the acceptance test words it; `the_same_prediction_runs_on_the_swapped_layer` stays pending (below).
  What holds (`memory_provider.rs`'s `a_prediction_is_duckdbs_but_for_its_drawn_examples`, on `base.binary` and `shape.same-table-twice.buyers`): every fetch planned as the typed request, no statement run; the description equal to DuckDB's; the prediction's result equal to DuckDB's; the materialized request and the model call body equal to DuckDB's but for `context`, the in-context examples.
- **A guard shows no domain crate changed for it**: `no_domain_crate_changed_for_t2_7` passes on this branch (no `T2.7` commit touches a crate `l0_deps.py` classes as domain); `git diff --stat d339e88c..HEAD` touches `sdm-testkit`, `sdm-host`, `rust/Cargo.lock`, the kit file and docs only.

### Probes

- The kit on `memory` (`--kit memory`): every case passes; `--fault truncate` and `--fault silent_capability` each exit 1.
- DuckDB against the provider on the swapped layer, `base.binary` and `shape.same-table-twice.buyers`, the observed run diffed field by field: the description, the request's task, schema, prediction rows and options, and the predicted result are equal; only `context`'s rows differ, and the `MISSING_INDICES` warning, which DuckDB's index advice gives and a provider with no index does not.
  The context differs because DuckDB's `USING SAMPLE reservoir(10000 ROWS) REPEATABLE (42)` returns even a table it keeps whole in a shuffled order (DuckDB 1.5.6: users 0 to 7 come back 6, 3, 4, 2, 7, 0, 1, 5), and the core's examples follow the sample's order.

### Deviations and questions

- **The swapped prediction's acceptance test cannot pass as written** (`the_same_prediction_runs_on_the_swapped_layer`, pending): it asks the swapped run's `core.materialized` and `core.predicted` to equal DuckDB's whole, including the in-context examples, which are a seeded sample each provider draws its own way (DuckDB's reservoir, SQLite's RS-2 stream, Postgres's `setseed`, the fallback's RS-1).
  Matching DuckDB's would mean reimplementing DuckDB's reservoir sampler in the test provider.
  Proposed to the test author: compare everything but `context` (as `memory_provider.rs` does), or compare the swapped run to the same provider on the default root, which is equal whole.
- **Found in the core, not changed here (a domain crate)**: `sdm-core/src/materialize.rs`'s `sampling_capabilities` refuses a provider that does not declare `time_bounds`, `sample_entities`, `neighbors` or `lookup` before it reads anything, even when the provider declares the in-memory fallback's limits, so a provider relying on the fallback cannot `materialize` (design choice 1 says such a request takes the fallback).
  This provider declares `time_bounds` and `sample_entities` itself instead; the core's check is for the orchestrator to route to its own package.
