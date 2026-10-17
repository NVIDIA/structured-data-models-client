# T2.3: Postgres

Lane c, on `t2.1b-on-21a` (T2.1b rebased onto T2.1a's approved head fdbd3265), not yet on a landed phase head.
The acceptance tests are the test author's (`T2.3 tests:` commits on `t2.3-tests`), run against this work from a scratch copy and kept out of this branch until the orchestrator combines the two.
Two commits of T2.2's (the runner's and comparator's database route, and the comparator's row-type vocabulary) are cherry-picked here at the orchestrator's request.

## What changed

- **The Postgres dialect** (`sdm-data/src/postgres.rs`): every request kind, every optional request declared, none falling back, every capability declared (`max_seed` any integer, `SeededStable`, `ValueByValue` text times, nulls last, no index advice).
  The catalog is read from `pg_catalog` in the contract's own shape, so the data layer reads it as a host provider's answer; a relation is found by `to_regclass` over its parts, each quoted alone, as a `FROM` finds the same name, one or two parts (rule `relation` otherwise).
  The preview, key counts, column ranges, repeated key and entity sample are the reference's statements but for the `FROM`; time bounds and windows compare `CAST(column AS timestamp)`, as on DuckDB.
  Sampling is the reference's `setseed` then `ORDER BY random() LIMIT n` (RS-6, KO2-24), the seed mapped by BLAKE2b-64 of its text as the reference maps it (`sdm-data/src/blake2b.rs`, RFC 7693, ported).
- **Key sets bound as the dialect binds them** (`P1 d:131`): one parameter, `$1`, a JSON array of rows of the cells' texts, unpacked by a common table expression named as the plan's relation.
  A number key is read as its own Postgres type and joined by value, so equal keys of any scale or integer type join and keys float64 would merge stay apart; a text key is read in the joined column's declared type (`jsonb_populate_record` over the table's row type), so a `uuid` or enum key compares as itself.
  Neighbors with a limit per seed read each seed's rows by `JOIN LATERAL`, as the reference does; the projected columns are renamed to positions inside the statement, so no source column meets a helper's name.
- **The psycopg provider** (`sdm_client._rust.providers.postgres`): runs the plan's statements on one cursor in a transaction of its own (a savepoint inside the caller's), binds the key set through psycopg's `%s` (its `$1` the only one, every `%` doubled), reads the result once (`fetchall`) and builds Arrow with pyarrow, no pandas, each column typed by its Postgres type's name in the driver's registry.
  `real` is read as the double of the server's text, as the reference reads it; `json` and `jsonb` as the server's text; `interval` as months, days and nanoseconds from the server's default style; `numeric(p,s)` as `decimal128` (`decimal256` past 38 digits), unconstrained `numeric` as text.
- **The core routes**: `_routing` gains the Postgres paths of `Schema.from_source`, `materialize` and `predict`, by backend (`FROM_SOURCE`, `MATERIALIZE`, `PREDICT`, `on_the_core`); the bridges name the provider and dialect of the source's backend.
  Each change to frozen bridge code is declared a refactor in `bridge_changes.json`; the routing seam's pinned `CORE_ROUTED` test names the Postgres paths.
- **rust-host's Postgres replay** (KO2-20, `sdm-testkit/src/replay.rs`): `run_python.py --record-fetches FILE` writes each case's fetches with the answers the Postgres provider replied with (an Arrow IPC stream in base64); `sdm-contract-run --replay FILE` runs each case on Postgres with every fetch answered from it, in order, a fetch other than the one recorded failing the run by name.
  Both markers list Postgres: `sdm_data::PROVIDERS` and `sdm_testkit::PROVIDERS`.
- **The comparator's Postgres row types** (`compare.postgres_arrow_name`): a psycopg row read's type names as the Arrow types the provider builds, at the orchestrator's request, with one test (`contract/tests/test_t23_row_types.py`).

## Probes

- All 86 cases on Postgres 17 (a local postgres:17 container) through python-core, compared with python-ref's committed lines (`compare.py --database postgres`): no difference in a fetch's answer to its own request; the differences are the statement counts (P2B-16) and the outcomes and payloads P2B-17, P2B-18 and P2B-19 register.
- rust-host replayed python-core's recording of all 86 cases: every fetch was the one recorded, and the cross-host core rule found no difference between the two hosts.
- The test author's tests, run from a scratch copy: `postgres_dialect.rs` 4 of 4, `test_the_postgres_provider.py` 6 of 6 (L4 executed on Postgres: every request kind on every base gives DuckDB's rows), `test_t23_postgres_keys.py` every pair-join test (lesson 19) passes, and the parity test fails on 106 vectors, all statement counts (P2B-16) or the exact decimals P2B-17 names.
- The client suites (`tests`, `tests_test_ops`) 2,839 passed with a Postgres server named.
- Name resolution, with `search_path = public, other`: a view is read on both routes, a table only in `other` is read on the core and refused on the old route (P2B-15), a name nowhere is refused `INVALID_REQUEST` on both.
- The replay's reader, outside JSON: 100,000 nested arrays are refused by name (serde_json's depth limit), an object of 300,000 members is read.
- The refusal census on Postgres (`evidence/census/postgres.json`), run with the census's `route` forcing each probe's routes on Postgres (`run_python.on_database`, a one-line local change the census needs as the runner did, not committed here): 71 probes, 25 departing sites, DuckDB's 23 of E92's class and two more, `rule:text_time` (P2B-19's class) and `rule:seed_size`, a seed past 2**64 - 1 (P2B-14, owner-pending, KO2-28).
- The rows-to-Arrow path: 200,000 rows of five columns, psycopg's read 0.6 to 0.7 s, the provider's whole read 0.9 to 2.0 s (`phase-1/dependencies.md`).
- The old route's from_source on Postgres crashed with `AttributeError` (`TimeColumnChoice.instead_of`) only in a venv holding a stale build of the PQL package; with the lane's packages it runs, and the S0 cases on Postgres pass.

## Deviations and what is left

- The statement counts (P2B-16, owner-pending): the reference reads `current_schema()` and counts every table's rows, reads it never uses; matching them means making those reads or a comparator change, the owner's choice.
- `key.decimal-38-10@postgres` differs only in P2B-1's `more_exact` fields, but the register's schema takes no database key there; a judging change.
- rust-host's Postgres lines carry no `driver` section, since no statement runs, and `compare.py`'s SQL rules ask one of every runner; a judging change (the SQL rules do not apply to a replay).
- The recording rust-host replays is 4.9 MB for all 86 cases; where it lives (a protected fixture the recorder makes, or a CI artifact of the Postgres `contract-python` job) is T2.3b's and the orchestrator's choice.
- H24's common-input comparison and its control moved to T2.1c (orchestrator); the first wave's Postgres legs (T2.17, T2.18, T2.8c, T2.19) are those packages'.
- The census's `route` forces each probe's DuckDB routes, as the runner did before T2.2's fix; the same one-line mapping is needed there (a judging change, T2.2's owner or the orchestrator's).
- `sdm-testkit`'s `t136_forecast` (`a_forecast_of_many_steps_is_refused_at_once`, then a two-second limit) failed once at 3.6 s with the machine's load at about 25 and passed alone; the base's 763b1b50 (T2.1b's tests) bounds it by its own run instead.
- Register IDs P2B-14 to P2B-19 are taken from the top of lane B's block, which T2.2 shares; the orchestrator renumbers if they meet.
