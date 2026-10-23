# sdm-data

The data layer: the only crate that writes SQL or knows a database dialect.
`Data` implements `sdm-contract`'s `DataLayer`: for each `DataRequest` it renders SQL statements in the source's dialect, or passes the request to a host provider, and reads each catalog answer into a dialect-neutral shape.
One private module per dialect: `duckdb.rs`, `sqlite.rs`, `postgres.rs`; `PROVIDERS` lists them.
Seeded sampling inside SQL matches the Python code with `mt19937.rs` (SQLite) and `blake2b.rs` (Postgres).
Depends only on `sdm-contract`, for the request and plan types.
It renders SQL and never runs it: the host runs the statements with its own driver.
Must not hold: a database driver, I/O, or schema, sampling or payload rules.
