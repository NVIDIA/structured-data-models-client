# sdm-sql

The SQL kit: the only crate that writes SQL or knows a database dialect.
A built-in source answers the core's typed requests through `kit::SqlKit`, over `Sql`: for each `DataRequest` it renders SQL statements in the source's dialect (`plan::SqlPlan`, for the source a host names by `plan::SqlChoice`), binds its key sets, has the host's driver run them, and reads each catalog answer into the standard shape every source answers in.
`Databases` implements `sdm-contract`'s `DataLayer` for the core: the shape a source of a built-in database answers in, and the semantic models it is read through, by the database the source declares.
The statement shapes are written once, in `sql.rs`, over the private `SqlDialect` trait; each dialect's private module (`duckdb.rs`, `sqlite.rs`, `postgres.rs`) implements it with only what differs: its quoting and relation depth, how a key set is named, bound and joined, its casts, the seeded sample, the per-seed neighbor read and the key lookup, and its catalog statements and their reading.
`sql::DIALECTS` is the one lookup by name, and `PROVIDERS` lists the same names; a new dialect adds one `impl SqlDialect` and one entry in each.
Seeded sampling inside SQL matches the Python code with `mt19937.rs` (SQLite) and `blake2b.rs` (Postgres).
It depends only on `sdm-contract`, for the request types.
It renders SQL and never runs it: the host runs the statements with its own driver.
Must not hold: a database driver, I/O, or schema, sampling or payload rules.

```text
used by:     sdm-host, (tests: sdm-core)
     |
sdm-sql
     |
depends on:  sdm-contract
```
