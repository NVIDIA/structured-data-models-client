# sdm-testkit (dev only)

A pure-Rust stand-in host for testing: it runs the contract cases in `contract/cases/` through `sdm-host` with no Python in the process.
Entry points: the `sdm-contract-run` program (`src/bin/`), the case loader (`case`), the host (`host`), the scripted model endpoint (`endpoint`), and test providers on DuckDB (`duck`), SQLite (`sqlite`), in-memory tables (`memory`) and recorded Postgres answers (`replay`).
Depends on `sdm-host` and `sdm-contract`, and on bundled DuckDB and SQLite builds.
Part of the migration's equivalence harness; it goes away with it.
Must not be a dependency of any shipped crate.
