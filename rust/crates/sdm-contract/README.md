# sdm-contract

The versioned messages and types that cross between the core and a host, and nothing else.
Main types: `ControlMessage` and `Step` (`run.rs`), the operations and their results (`ops.rs`), `DataRequest`, `FetchPlan` and the `DataLayer` trait (`data.rs`), `CoreError` and `ErrorCode` (`error.rs`), `HostFacts`, `ColumnType` and `Scalar`.
`schema_json()` writes the JSON Schema of every message; the files under `schema/` and `schema/contract.lock` hold it, and a change of shape or meaning bumps `CONTRACT_VERSION`.
Depends on no crate of ours, only on Arrow types, serde and schemars, so every other crate and every host can share it.
Feature `test-ops` adds operations only tests reach; `schema-check` checks JSON against the schema in tests.
Must not hold: logic, I/O, SQL, or anything that decides an answer; those belong in the crates that use these types.
