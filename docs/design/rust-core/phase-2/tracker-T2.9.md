# Phase 2 tracker: T2.9

The design branch owns `tracker.md`; this file holds T2.9's section until it folds in.
Lane b, on the phase head after T2.1j's landing (c8075715): this package's `T2.9:` commits, the test author's (`T2.9 tests:`, cherry-picked from `t29-tests`), then the baselines, goldens and pending list.

## T2.9

T2.9 is forecasting's options on top of the split already in the core (E153): `num_forecasts`, `lag_timesteps`, quantiles, and `response_field` for quantiles, on both hosts and DuckDB, SQLite and Postgres.

### What changed

- **Quantiles** (`39fd07fe`). `sdm-wire` reads an item's `quantiles` as the bindings' `_float_dict` reads `probabilities` (refusals name `prediction.quantiles`), so the core's `response_field` refusal of it is gone.
  `sdm-request`'s layout keeps each answer's quantiles through a merge of calls and a forecast's steps; the result's `quantiles` column is a map of each quantile (utf8, as the endpoint named it) to its value (float64), after `probabilities`; `absent()` reports `quantiles` only where no answer holds any.
  The Python host hands each row's quantiles to `PredictionItem`, so `to_pandas()` spreads them as `quantile_<key>`, as on the old route.
- **`lag_timesteps`** (`9e1deb95`). The value travels in `Materialize.values` (contract v27) and is read by `sdm-request`'s `AskedLag` as the reference's engine meets it: negative, or not ordered against zero (text, null, a list), refused for any query; not whole (a float, `true`) or past `u64` refused for a temporal query; otherwise the steps a temporal query adds, none for a static one (which the reference takes and ignores).
  `sdm-core` refuses it where the engine does, once the context is labelled, so the statements before the refusal are the reference's.
  For a temporal query, `sdm-core/src/lag.rs` moves each example's anchor back by the steps' windows (a month's by the calendar, as pandas multiplies a `DateOffset`), reads the aggregated table's rows from there in one `_by_time` read (the context's examples before narrowing, then the entities predicted for, as the reference's `sample_lagged_target`), and lays out `__kumo_arl0__` to `__kumo_arl<n-1>__`, each the target labelled by `sdm_engine::labels` over the window that many steps back.
  `sdm-request`'s payload puts them after the label in both halves with no semantic type: an aggregation's `float64`, a condition's `bool` (`float64` where no example has one).
  A payload the lags would make larger than the endpoint takes is refused before any column is computed.
- **Contract v27** (`9e1deb95`), the lock's working entry beside the merged v26: `Materialize.values` carries `lag_timesteps`; it leaves `Materialize.options`' names; `Predicted.result`'s `quantiles` column is the map above, in place of v26's list of float structs, which could neither keep a key as written (`"0.50"`) nor hold one that is not a number, both of which the reference's `quantile_<key>` columns keep.
  `messages.py` writes 27; the v26 vector tests read their vectors under v27.
- **`num_forecasts`**: no product change; E153's split answers every `num_forecasts` case, now with lags and quantiles per step.
- **rust-host** (`76172f28`, protected `sdm-testkit/src/host.rs`): a case's `lag_timesteps` goes into the call's values as written, predict asks the case's `outputs`, and the scripted endpoint answers probabilities and quantiles where the case asks.
- **Judged data**: the census's `rule:lag_timesteps` probe taken out (its site is gone); bridge changes declared `moved` (`_rust/materialize.py::<module>`, `_rust/predict.py::predict`); baselines for the 26 cases on all three runners (`ratchet.py accept --checkpoint c8075715`); 17 L4 goldens per database.

### Probes

- **Differential, old route against core route** (`t29-impl/diff_lag.py` in the session scratchpad, not committed): 40 generated DuckDB sources (users 3 to 60, orders 0 to 900, values DOUBLE, BIGINT, INTEGER or FLOAT with nulls, steps of 1 to 13 days), twelve queries each (SUM, COUNT, AVG, MIN, MAX, three comparisons, `= NULL`, a window not starting at 0, FORECAST 2, a static target), windows of days, weeks, months and hours, lags 1 to 3, anchors given or not, `num_context_samples` 5, 17 or 50; plus each value class on the shop base. 504 calls, 449 answered: every payload equal to the reference's.
  The 40 outcomes that differ are `MAX(...) = NULL` (P1B-21, the core's `null_compared`, unchanged by T2.9) and the two overflowing lags (P2B-18).
  An earlier run with `DECIMAL(10,2)` values showed the lag columns' sums exact where the reference sums decimals as float64 first: the class T2.0c made the core's (more exact, registered per case under KO2-50's rule); no T2.9 case has a decimal target.
- **The cases**: all 26 on python-core (DuckDB, SQLite, local Postgres 17), rust-host (DuckDB, SQLite) and python-old (DuckDB) against python-ref's lines: `compare.py` finds no difference (diagnostics only: SQL text and bindings).
- **The test author's tests**: all 15 of `test_t29_forecasting.py` (three databases) and all 8 of `t29_forecasting.rs` pass; the pending list is empty.
- **Units**: `AskedLag` by value and query kind; the payload's lag columns and dtypes against the locked payload schema; the lag's windows, alignment, months and limits (`lag.rs`); quantiles laid out and malformed ones refused (`sdm-core/tests/predict.rs`); the host route's lag request equal to the old route's for a regression, a binary task, an average and a static target.

### Deviations and questions

- **An average, a minimum or a maximum's lag is the reference's, misaligned** (matched, a question for the orchestrator).
  The reference's executor labels a lag through `execute_join`, which drops the examples whose window holds no value, and the engine's frame of the lags lines the rest up by position, so each value moves to the first examples rather than its own, and the last examples get none (shown: user 0's predicted row at 2023-08-01 gets no lag where its window holds 112.0).
  The core matches it (`lag.rs`, its module docs), so the payloads are equal.
  Fixing it (each value on its own example) is one line in `Lag::columns` and a register entry; it changes what the model is told, so it is the orchestrator's to decide (KO2-70).
- **P2B-18** (proposed, approver reviewer): a lag reaching outside what a time holds, or past `u64`, is refused `VALUE_OUT_OF_RANGE` naming `lag_timesteps`, where the reference's engine overflows and its client says `INTERNAL_ERROR`.
- **P2B-19** (proposed, approver owner, so the orchestrator under KO2-70): a NaN `lag_timesteps` is refused `INVALID_REQUEST` on the core route (the call's values travel as JSON), where the reference takes it and adds no lag; diagnostic test `test_a_nan_lag_is_refused_on_the_core_route_only`.
- **Contract v27** is opened here; T2.8 and T2.12 also carry v26 edits, so whichever lands second regenerates the schema and lock entry on rebase (mechanical).
- **Judging paths touched**: `sdm-testkit/src/host.rs`, `scripts/rust_core/bridge_changes.json` and `bridge_freeze.json`, `contract/refusal-census.json`, `contract/baselines/**`, `rust/crates/sdm-contract/schema/**`, `contract/tests/t29-pending.json` (pending only); none of `compare.py`, `run_python.py`, `ratchet.py`, `gate.sh` or `ci.yml`.
- `tests/test_t21b_contract_v26.py` and `sdm-contract/tests/contract_26_vectors.rs` now hold v26 as locked and the core at v26 or later, rather than v26 as the newest.
