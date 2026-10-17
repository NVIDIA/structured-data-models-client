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
- **rust-host's Postgres replay** (KO2-20, `sdm-testkit/src/replay.rs`): `run_python.py --record-fetches FILE` writes each case's fetches with the answers the Postgres provider replied with (an Arrow IPC stream in base64) and the text of each statement the driver ran for it, from python-core's driver capture; `sdm-contract-run --replay FILE` runs each case on Postgres with every fetch answered from it, in order, a fetch other than the one recorded failing the run by name, and records those statements as run, its query log the recorded texts in order and each result its own read of the answer replayed, so its driver record is complete with no comparator change (the orchestrator's answer).
  The recording is made on the fly: in CI an artifact of the Postgres python-core job handed to rust-host's (T2.3b), locally by running python-core with `--record-fetches`; it is never committed (the orchestrator's answer).
  Both markers list Postgres: `sdm_data::PROVIDERS` and `sdm_testkit::PROVIDERS`.
- **The comparator's Postgres row types** (`compare.postgres_arrow_name`): a psycopg row read's type names as the Arrow types the provider builds, at the orchestrator's request, with one test (`contract/tests/test_t23_row_types.py`).

## Probes

- All 86 cases on Postgres 17 (a local postgres:17 container) through python-core, compared with python-ref's committed lines (`compare.py --database postgres`): no difference in a fetch's answer to its own request; the differences are the statement counts (P2C-2) and the outcomes and payloads P2C-3, P2C-4 and P2C-5 register.
- rust-host replayed python-core's recording of all 86 cases: every fetch was the one recorded, and the cross-host core rule found no difference between the two hosts.
  With the replayed statements recorded, rust-host's statement counts are python-core's, and python-core against rust-host differs in 4 statement results, all a row read's representation: `timestamptz` has no Arrow name (its zone is the session's, which the type name does not say) on `time.timestamptz@postgres` (3), and an interval's row value (a list) against its Arrow value (`interval`) on `time.feature-interval@postgres` (1).
- The test author's tests, run from a scratch copy: `postgres_dialect.rs` 4 of 4, `test_the_postgres_provider.py` 6 of 6 (L4 executed on Postgres: every request kind on every base gives DuckDB's rows), `test_t23_postgres_keys.py` every pair-join test (lesson 19) passes, and the parity test fails on 106 vectors, all statement counts (P2C-2) or the exact decimals P2C-3 names.
- The client suites (`tests`, `tests_test_ops`) 2,839 passed with a Postgres server named.
- Name resolution, with `search_path = public, other`: a view is read on both routes, a table only in `other` is read on the core and refused on the old route (P2C-1), a name nowhere is refused `INVALID_REQUEST` on both.
- The replay's reader, outside JSON: 100,000 nested arrays are refused by name (serde_json's depth limit), an object of 300,000 members is read.
- The refusal census on Postgres (`evidence/census/postgres.json`), run with the census's `route` forcing each probe's routes on Postgres (`run_python.on_database`, a one-line local change the census needs as the runner did, not committed here): 71 probes, 25 departing sites, DuckDB's 23 of E92's class and two more, `rule:text_time` (P2C-5's class) and `rule:seed_size`, a seed past 2**64 - 1 (P2C-6, owner-pending, KO2-28).
- The rows-to-Arrow path: 200,000 rows of five columns, psycopg's read 0.6 to 0.7 s, the provider's whole read 0.9 to 2.0 s (`phase-1/dependencies.md`).
- The old route's from_source on Postgres crashed with `AttributeError` (`TimeColumnChoice.instead_of`) only in a venv holding a stale build of the PQL package; with the lane's packages it runs, and the S0 cases on Postgres pass.

## Deviations and what is left

- The statement counts (P2C-2, owner-pending): the reference reads `current_schema()` and counts every table's rows, reads it never uses; matching them means making those reads or a comparator change, the owner's choice.
- `key.decimal-38-10@postgres` differs only in P2B-1's `more_exact` fields; the register's schema takes no database key there, which the orchestrator fixes in a judging landing before T2.3, after which P2C-3 lists them.
- The two row-read representations above (`timestamptz`'s zone, an interval's row value) are comparator vocabulary for the orchestrator to route.
- KO2-20 as built replays the Postgres server's answers to the core's own statements, not the reference's (whose statements differ and whose capture keeps digests only); the orchestrator agrees and tells the owner.
- Owner rows: P2C-3 cites KO2-35 and P2C-4 E66 (as P1B-11 and P1B-8); P2C-1 (a name found on the search path) and P2C-5 (text times on Postgres) have no owner row, and P2C-2 and P2C-6 are with the owner.
- H24's common-input comparison and its control moved to T2.1c (orchestrator); the first wave's Postgres legs (T2.17, T2.18, T2.8c, T2.19) are those packages'.
- The refusal census is now a one-time report at the phase's end, not a gate step (the owner); its `route` would need the runner's database mapping then, and the evidence above stays as a record.
- `sdm-testkit`'s `t136_forecast` (`a_forecast_of_many_steps_is_refused_at_once`, then a two-second limit) failed once at 3.6 s with the machine's load at about 25 and passed alone; the base's 763b1b50 (T2.1b's tests) bounds it by its own run instead.
- Register IDs are lane C's (P2C-1 to P2C-6), under a lane C banner (the orchestrator: T2.3 is lane C).
