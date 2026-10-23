# sdm-core

The run state machine and the operations a host can start.
Entry points: `start` (`dispatch.rs`) begins an operation and returns a `Run`; the host asks the run for its next `Step`, answers it with a `Reply`, and repeats until `Done`.
Operations: describe a source (`describe.rs`), edit a schema (`edit.rs`), validate a query (`pql.rs`), materialize (`materialize.rs`, `subgraph.rs`, `fallback.rs` for in-memory sampling), predict (`predict.rs`, `ranking.rs`).
`CompositionRoot` holds the run's data layer; `profiles.rs` checks what the chosen model can answer.
Depends on `sdm-pql` (only through `pql.rs`), `sdm-schema`, `sdm-engine`, `sdm-request`, `sdm-wire` and `sdm-contract`.
It reaches data only through the `DataLayer` trait, so it does not depend on `sdm-data`.
Must not hold: SQL, a dialect's name, I/O, or anything specific to one host language.
