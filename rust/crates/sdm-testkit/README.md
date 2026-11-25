# sdm-testkit (dev only)

A pure-Rust stand-in host for testing: it runs the contract cases in `contract/cases/` through `sdm-host` with no Python in the process.
Entry points: the `sdm-contract-run` program (`src/bin/`), the case loader (`case`), the host (`host`), the scripted model endpoint (`endpoint`), and test providers on DuckDB (`duck`), SQLite (`sqlite`), in-memory tables (`memory`) and recorded Postgres answers (`replay`).
It depends on `sdm-host` and `sdm-contract`, and, among its third-party dependencies, on bundled DuckDB and SQLite builds.
Maintained correctness harness for native execution, provider contracts and error handling.
Must not be a dependency of any shipped crate.

```text
used by:     no crate of ours
     |
sdm-testkit
     |
depends on:  sdm-host, sdm-contract
tests also:  sdm-host built with test-ops
```
