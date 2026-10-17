# T2.3: Postgres

Lane c, on `t2.1d` (T2.1d, the judging-path changes T2.3 needs, on T2.1b's head on lane d, 60b07bbe); both are rebased when T2.1b lands.
T2.3 edits no judging file: the runner's recording, rust-host's replay, the comparator's Postgres row types, an interval's canonical cell and `more_exact`'s database keys are T2.1d's (the orchestrator's decision).
The acceptance tests are the test author's (`T2.3 tests:` commits on `t2.3-tests`), run against this work from a scratch copy and kept out of this branch until the orchestrator combines the two.

## What changed

- **The Postgres dialect** (`sdm-data/src/postgres.rs`): every request kind, every optional request declared, none falling back, every capability declared (`max_seed` 2**64 - 1, `SeededStable`, `ValueByValue` text times, nulls last, no index advice).
  The catalog is read from `pg_catalog` in the contract's own shape, so the data layer reads it as a host provider's answer, one or two parts (rule `relation` otherwise).
  A relation is found as the reference finds a table: a one-part name in `current_schema()` alone, two parts by schema and table; a one-part name a temporary or `pg_catalog` relation shadows answers no columns, so no read of a run reads a relation the reference does not (P2C-1).
  The preview, key counts, column ranges, repeated key and entity sample are the reference's statements but for the `FROM`.
  A time column is compared, ordered and bounded as the reference does it, in its own type (a text column by its text): the window compares the column with the seed's time read in the column's declared type, the neighbor order is the column itself, and the bounds are `MIN` and `MAX` of the column, then read as timestamps, which the request takes back.
  Sampling is the reference's `setseed` then `ORDER BY random() LIMIT n` (RS-6, KO2-24), the seed mapped by BLAKE2b-64 of its text as the reference maps it (`sdm-data/src/blake2b.rs`, RFC 7693, ported).
- **Key sets bound as the dialect binds them** (`P1 d:131`): one parameter, `$1`, a JSON array of rows of the cells' texts, unpacked by a common table expression named as the plan's relation.
  A number key is read as its own Postgres type and joined by value, so equal keys of any scale or integer type join and keys float64 would merge stay apart; a text key, and a seed time, are read in the joined column's declared type (`jsonb_populate_record` over the table's row type), so a `uuid` key compares as itself and a text time as the reference's text.
  Neighbors with a limit per seed read each seed's rows by `JOIN LATERAL`, as the reference does; the projected columns are renamed to positions inside the statement, so no source column meets a helper's name.
- **The psycopg provider** (`sdm_client._rust.providers.postgres`): runs the plan's statements on one cursor in a transaction of its own (a savepoint inside the caller's), binds the key set through psycopg's `%s` (its `$1` the only one, every `%` doubled), reads the result once (`fetchall`) and builds Arrow with pyarrow, no pandas, each column typed by its Postgres type's name in the driver's registry.
  `real` is read as the double of the server's text, as the reference reads it; `json` and `jsonb` as the server's text; `interval` as months, days and nanoseconds (pyarrow's `MonthDayNano`); `timestamptz` as the instant in UTC whatever the session's zone; `numeric(p,s)` as `decimal128` (`decimal256` past 38 digits), unconstrained `numeric` as text.
  A seed time is written as pandas' `isoformat` writes it, the text the reference binds.
- **The core routes**: `_routing` gains the Postgres paths of `Schema.from_source`, `materialize` and `predict`, by backend (`FROM_SOURCE`, `MATERIALIZE`, `PREDICT`, `on_the_core`); the bridges name the provider and dialect of the source's backend.
  Each change to frozen bridge code is declared a refactor in `bridge_changes.json`; the routing seam's pinned `CORE_ROUTED` test names the Postgres paths.

## Probes

- All 86 cases on Postgres 17 (a local postgres:17 container), python-core recording its fetches and rust-host replaying them (T2.1d), compared with python-ref's committed lines: no statement's answer differs, rust-host's core fields and statement results equal python-core's, and every difference left is registered: the statement counts (P2C-2), exact decimals (P2C-3), two refusals named as on DuckDB (P2C-4) and `time.text-one-bad-value`'s code (P2C-5).
  `time.text-us-date` now answers as the reference answers, its windows by the column's text.
- The test author's tests, from a scratch copy: `postgres_dialect.rs` 4 of 4; `test_the_postgres_provider.py` 6 of 6 (L4 executed on Postgres); `test_t23_postgres_keys.py` 92 passed, every pair-join test (lesson 19) among them, and 106 parity failures, all statement counts (P2C-2) or the 19 exact-decimal vectors P2C-3 names.
- Name resolution, `search_path = public, other`: a view is read on both routes; a table only in `other` and a name nowhere are refused `INVALID_REQUEST` on both; a public table shadowed by a temporary table of its name is read on the old route and refused `INVALID_REQUEST` on the core (P2C-1).
- The refusal census on Postgres (`evidence/census/postgres.json`, a record; the census is now a one-time report at the phase's end, the owner): 71 probes, 25 departing sites, DuckDB's 23 of E92's class, `rule:text_time` (P2C-5) and `rule:seed_size` (P2C-6).
- The rows-to-Arrow path: 200,000 rows of five columns, psycopg's read 0.6 to 0.7 s, the provider's whole read 0.9 to 2.0 s (`phase-1/dependencies.md`).
- The client suites (`tests`, `tests_test_ops`): 2,839 passed with a Postgres server named, before the move onto T2.1d; since then the provider and routing tests, focused.

## Deviations and what is left

- The statement counts (P2C-2, owner-pending): the reference reads `current_schema()` and counts every table's rows, reads it never uses; no wasted query is added meanwhile.
- P2C-1 (owner-pending): a one-part name a temporary or `pg_catalog` relation shadows is refused by name where the reference reads its schema's table; a plan would have to name `current_schema()`'s schema, which the core does not know when it plans.
- P2C-5: a text time value Postgres cannot read as a timestamp fails the bounds as the provider's error; `time.text-one-bad-value`, where the reference refuses by name too, is E66's named refusal; the census's `rule:text_time` probe, where the reference answers, needs the bounds as text, which the contract's `time_bounds` answer does not carry (owner-pending under KO2-28).
- P2C-6 (owner-pending): a seed past 2**64 - 1, which `max_seed`'s type cannot declare.
- KO2-20 as built replays the Postgres server's answers to the core's own statements, not the reference's (whose statements differ and whose capture keeps digests only); the orchestrator agrees and tells the owner.
- H24's common-input comparison and its control moved to T2.1c (orchestrator); the first wave's Postgres legs (T2.17, T2.18, T2.8c, T2.19) are those packages'.
- `sdm-testkit`'s `t136_forecast` once failed its old two-second limit under a load of about 25; the base's T2.1b test now bounds it by its own run.
- Register IDs are lane C's (P2C-1 to P2C-6), under a lane C banner.
