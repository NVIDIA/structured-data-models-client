# sdm-contract

The versioned messages and types that cross between the core and a host, and nothing else.
Main types: `ControlMessage` and `Step` (`run.rs`), the operations and their results, including the described schema `SourceDescription` (`ops.rs`), `DataRequest`, `FetchPlan` and the `DataLayer` trait (`data.rs`), `CoreError` and `ErrorCode` (`error.rs`), `HostFacts`, `ColumnType` and `Scalar`.
`schema_json()` writes the JSON Schema of every message; the files under `schema/` and `schema/contract.lock` hold it, and a change of shape or meaning bumps `CONTRACT_VERSION`.
It depends on no crate of ours, only on third-party crates (`arrow-array`, `arrow-buffer`, `arrow-schema`, `base64`, `schemars`, `serde`, `serde_json`, `serde_stacker`), so every other crate and every host can share it.
Feature `test-ops` adds operations only tests reach; `schema-check` checks JSON against the schema in tests.
Must not hold: logic, I/O, SQL, or anything that decides an answer; those belong in the crates that use these types.

```text
used by:     sdm-wire, sdm-schema, sdm-data, sdm-engine, sdm-request, sdm-core, sdm-host, sdm-py, sdm-testkit
     |
sdm-contract
     |
depends on:  no crate of ours
```
