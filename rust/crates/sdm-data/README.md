# sdm-data

The data layer: the only crate that writes SQL or knows a database dialect.
`Data` implements `sdm-contract`'s `DataLayer`: for each `DataRequest` it renders SQL statements in the source's dialect, or passes the request to a host provider, and reads each catalog answer into a dialect-neutral shape.
The statement shapes are written once, in `sql.rs`, over the private `SqlDialect` trait; each dialect's private module (`duckdb.rs`, `sqlite.rs`, `postgres.rs`) implements it with only what differs: its quoting and relation depth, how a key set is named, bound and joined, its casts, the seeded sample, the per-seed neighbor read and the key lookup, and its catalog statements and their reading.
`sql::DIALECTS` is the one lookup by name, and `PROVIDERS` lists the same names; a new dialect adds one `impl SqlDialect` and one entry in each.
Seeded sampling inside SQL matches the Python code with `mt19937.rs` (SQLite) and `blake2b.rs` (Postgres).
It depends only on `sdm-contract`, for the request and plan types.
It renders SQL and never runs it: the host runs the statements with its own driver.
Must not hold: a database driver, I/O, or schema, sampling or payload rules.

```text
used by:     sdm-host, (tests: sdm-core)
     |
sdm-data
     |
depends on:  sdm-contract
```
