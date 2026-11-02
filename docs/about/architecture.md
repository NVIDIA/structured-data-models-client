<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Architecture

Python and Java expose the client APIs; a shared Rust core validates queries, constructs model requests and interprets results.
The model endpoint performs inference.
Installing the client does not download model weights or require a GPU.

## Request flow

```text
Python frame/array inputs                Java Arrow / source objects
             |                                        |
             +---------- public client APIs ----------+
                                   |
                              PyO3 / JNI
                                   |
                    Rust query and request operations
                      /                            \
             in-memory sampling               SQL sampling plan
                      |                            |
                      |                  host executes on connection
                      |                            |
                      +------ model request -------+
                                   |
                       host sends HTTP to endpoint
                                   |
                       Rust interprets the response
                                   |
                      Python / Java prediction result
```

Accepted input types depend on the public API entrypoint.
Host adapters own language objects, credentials, connections and network I/O.
Rust owns shared validation, sampling, SQL generation and request/result transformations.
The internal protocol lets Rust request host I/O without embedding Python or Java in the domain crates.

## Public operations

| Operation | What it does |
|---|---|
| `Schema.from_data` / Java `Schema.fromData` | Describe in-memory tables, their columns, keys and relationships |
| `Schema.from_source` / Java `Schema.fromSource` | Describe selected tables in a connected source |
| `PredictiveQuery.materialize` | Validate a question and gather context plus rows to score for the selected model |
| Direct `Request` | Supply context and prediction data directly for Tabular |
| `client.predict` | Send a stateless request and return interpreted predictions |
| `client.fit` | Pin context in a model endpoint session; this does not train model weights |
| Fitted handle `predict` / `close` | Score against pinned context, then release the session |

A request is prepared for one model and must be sent to that model.
Tabular uses a single table; Relational supports linked tables and relational queries.
For a Tabular table stored externally, read it into supported in-memory input first; source-backed schema sampling belongs to Relational.
Session support also depends on the deployment; a transport that cannot serve sessions reports `SESSIONS_UNSUPPORTED`.
See the [quickstart](../get-started/quickstart.md), [prediction output](../reference/prediction-output.md) and [public API contract](../reference/stability.md) for user-facing details.

## Packages and boundaries

| Component | Owns |
|---|---|
| Python `structured-data-models-client` (`sdm_client`) | Public API, host adapters, HTTP transport and bundled native binding |
| Python `structured-data-models-connectors` (`sdm_connectors`) | Source objects, drivers, readers and connection lifecycle |
| Python `structured-data-models-api` (`sdm_api`) | Generated model endpoint wire types |
| Java `com.nvidia.sdm` | Public Java API, Arrow conversion, source/HTTP adapters and JNI integration |
| `sdm-host`, `sdm-contract` | Native handles, operation protocol and typed data exchange |
| `sdm-core` | Coordinate shared operations |
| `sdm-pql`, `sdm-schema` | Query parsing/validation and schema rules |
| `sdm-engine`, `sdm-request`, `sdm-wire` | Sampling, request/session computation and model payload/result handling |
| `sdm-data` | SQL generation and dialect-specific catalog interpretation |
| `sdm-testkit` | Test host and provider checks; not a runtime dependency |

Only the language adapters execute database and HTTP operations.
SQL dialect logic stays in `sdm-data`; changing in-memory sampling must not duplicate warehouse query logic in a host adapter.
The native protocol is versioned for matching bindings, not a separately supported public API.
See [Rust crate rules](../../rust/README.md) for the enforced dependency boundaries.

## Compatibility and validation

The [compatibility ledger](../../tests/parity/README.md) maps supported APIs and capabilities across languages, models, connectors and type families.
API snapshots and behavior tests enforce those declarations, including explicit applicability exceptions.
Local contract tests do not establish live endpoint coverage; endpoint evidence and final runtime/performance acceptance remain separate.
Maintained tests use the current client and self-contained fixtures, without installing a historical implementation.
