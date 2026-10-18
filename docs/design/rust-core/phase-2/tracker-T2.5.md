# Phase 2 tracker: T2.5

The design branch owns `tracker.md`; this file holds T2.5's section until it folds in.
Lane c, on the phase head after T2.18's landing (f42d92ea), then the test author's acceptance tests (`T2.5 tests:`, cherry-picked from `t25-tests`), then this package (`T2.5:` commits).
T2.5 and T2.6 are one package (the orchestrator's decision): T2.6's exits are kit cases.

## T2.5

T2.5 is the provider conformance kit and the in-memory fallback: one fixture dataset and one expected-answers file under `contract/fixtures/provider-kit/` (the test author's), run through the core on every landed provider by a runner on each host, and the fallback that answers an optional request a provider does not declare, within limits that fail loudly by name.

### What changed

- **The in-memory fallback** (`sdm-core/src/fallback.rs`, new): a provider that declares `ProviderCapabilities::fallback` (contract v26, T2.1b) has each optional request it does not declare answered by the core from `scan`s (`DataAccess::route`, `Asker::fetch`), in the shape the request's pushdown gives.
  Every optional request is answered: `preview`, `count_rows`, `key_counts`, `time_bounds`, `column_ranges`, `repeated_key`, `sample_entities`, `neighbors` (window, order, row limit per seed) and `lookup`.
  Ordering, grouping and distinct counts use Arrow's row format (`arrow-row`, a new dependency of `sdm-core`, already in the lockfile); a sample is the registered stream (RS-1) over the rows in that order, so the same rows on every provider.
  Limits: each scan asks for one row past what is left of `max_rows`, so no scan moves more; past `max_rows`, or past `max_bytes` of Arrow data held (seeds, scans and the answer), the run ends with `UNSUPPORTED_REQUEST`, fields `request`, `rule` `fallback_limit`, `limit` and `most`.
  What it does not read (a time in a column that is neither a zone-less timestamp nor a date, a key joined to seeds of another type but integers of two widths, a type the row format does not order) ends the run the same way, rule `fallback`, naming the field.
  A provider's failure of a scan is the request's failure; a declared request is never the fallback's.
- **The test-only fetch** (`sdm_contract::test_ops::TestFetch`): `undeclared` and `fallback`, so a kit case takes a capability away from any provider, a SQL one included, and declares the fallback; test-only, so no public API and no contract change.
- **The Python run loop** (`_rust/run.py`): a provider may answer a fetch with a `Page` (data and a continuation), which the loop hands to the core with its continuation; a provider's exception carrying `retryable` as `True` reaches the core retryable, where before every provider error was not.
- **The kit's runners**: `contract/runners/provider_kit.py` (Python host: DuckDB, SQLite and Postgres providers through the run loop) and `sdm-contract-run --kit` (`sdm_testkit::kit`: the DuckDB and `rusqlite` providers), each writing the report the test author's tests read and judging it by `expected.json`'s rules.
  Each pages every answer in pages of `page_rows`, fails an injected page, and seeds the faults `truncate` and `silent_capability` in its provider path.
  The Rust runner reads the test-only start and result as JSON, so it builds without the test-only operations; `cargo test` builds it with them.
- **CI** (`ci.yml`): the three `contract-python (ref-pandas3, <database>)` jobs take the test-ops wheel too and install it alone in an interpreter over the job's environment (`venv --system-site-packages`), named by `SDM_KIT_PYTHON`; every step still runs the shipped build, and the kit's runner, started on it, runs itself again under that interpreter.
  The DuckDB job runs `test_t25_provider_kit.py` in its `contract/tests` run; the SQLite and Postgres jobs run it for their database (`-k "$DATABASE"`).
- **Carried rows**: G5, rust-host runs a case's `validate_query` before `materialize` on DuckDB too, and stops at a refusal (`host.rs`; test `validate_before_materialize.rs`); M9, rust-host holds a policy refusal's `field` and `reason` where the policy names them (`t128b_predict_result.rs`).
- **Docs**: three rows of `phase-1/dependencies.md` (choice 9): paging, `arrow-row`, the fallback's sample.

### Probes

- The test author's tests: all 19 of `test_t25_provider_kit.py` pass on DuckDB, SQLite and Postgres (a local Postgres 17), and all 12 of `t25_provider_kit.rs`; the pending list is empty.
- The fallback's own tests (`sdm-core/tests/fallback.rs`, 15): each request's answer worked out by hand, the limits by name, the same sample whatever order the provider scans in, integer keys of two widths joined exactly, a provider's scan failure carried with its retryability, a declared request never the fallback's.
- The kit's test file run from a shipped build of `sdm_client._native` with `SDM_KIT_PYTHON` naming a test-ops interpreter: all 19 pass; with none named, the runner exits 2, naming the build.
- Before handoff: the cheap prepush tier, `cargo test --workspace --no-fail-fast` (137 test targets), `pytest contract/tests -n 4 --dist loadfile` with a Postgres server named (4,669 passed), `tests/test_ci_workflow.py`, `tests/test_wheel_smoke.py`, `tests/test_release_contract.py`, `register.py validate` and `ratchet.py check --checkpoint f42d92ea` (17 protected paths for the landing to name): all pass.
- Measured, a release build over 1,000,000 rows: `neighbors` (window, five per seed, 1,000 seeds) 0.32 s, `key_counts` 0.31 s.

### Deviations and questions

- **The kit's Python run needs the test-ops build** (the orchestrator's open point 2): the shipped wheel has no single-request entry point and none was added; the runner hands itself to `SDM_KIT_PYTHON`'s interpreter (above), no test is left out of any run (`l0_scan.py` bans that), and the shipped build stays under every other step.
  Locally, `pytest contract/tests` runs it on the test-ops build CONTRIBUTING.md installs; on a shipped build with no `SDM_KIT_PYTHON` its runner exits 2, naming the build.
- **Left out of the kit** (the orchestrator's open point 4): `time_bounds`, `column_ranges`, `sample_entities` and the catalog reads; the fallback answers the first three, held by its own tests only.
- **The fallback reads no text time** (SQLite stores times as text): a `time_bounds`, window or time order over a text column ends the run, rule `fallback`; Phase 4's frames carry typed times.
- **rust-host Postgres** is a replay, not a provider: the Python side runs the kit there (the orchestrator agreed).
- **M9** has no rust-host case today whose policy names a field or reason (only the Databricks serving cases do, which run on python-core), so the check holds none yet.
