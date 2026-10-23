# sdm-core

The run state machine and the operations a host can start.
Entry points: `start` (`dispatch.rs`) begins an operation and returns a `Run`; the host asks the run for its next `Step`, answers it with a `Reply`, and repeats until `Done`.
Operations: describe a source (`describe.rs`), edit a schema (`edit.rs`), validate a query (`pql.rs`), materialize (`materialize.rs`, `subgraph.rs`, `fallback.rs` for in-memory sampling), predict (`predict.rs`, `ranking.rs`).
`CompositionRoot` holds the run's data layer; `profiles.rs` checks what the chosen model can answer.
It depends on `sdm-schema`, `sdm-engine`, `sdm-request`, `sdm-wire`, `sdm-contract`, and `sdm-pql`, which only `pql.rs` and the test-only `test_ops.rs` call.
Shipped code reaches data only through `sdm-contract`'s `DataLayer` trait; `sdm-data` is a test-only dependency, used to build a real data layer in `tests/`.
Must not hold: SQL, a dialect's name, I/O, or anything specific to one host language.

```text
used by:     sdm-host
     |
sdm-core
     |
depends on:  sdm-pql, sdm-schema, sdm-engine, sdm-request, sdm-wire, sdm-contract
tests also:  sdm-data
```
