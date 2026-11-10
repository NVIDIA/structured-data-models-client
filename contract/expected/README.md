<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Current native outcomes

These are explicit regression expectations, not outputs automatically accepted from a test run.
`check_observations.py` also checks complete case inventory, input identity, native operations, SQL capture and endpoint payloads.
Do not update expectations just to make a run pass; establish whether the public contract or the test is wrong.

`duckdb-python.json` retains the previous native outcomes for 309 cases, with these six reconciliations:

| Cases | Expected behavior | Existing basis |
|---|---|---|
| `s0.boundary.date-packed`, `date-long-field` | Reject at materialize with `INVALID_REQUEST` | Rust `sdm-testkit/tests/s0_cases.rs` explicitly rejects packed date literals |
| `s0.boundary.depth-999` | Accept at materialize | Native parser accepts depth 999; the deleted Python shadow parser caused the old recursion failure |
| `s0.boundary.escape-lone-surrogate`, `escape-surrogate-pair` | Accept at materialize | Native parser replaces lone surrogates and joins pairs; existing PQL-1 decision and Rust S0 checks |
| `target.aggregation.list-distinct` | Reject at materialize with `UNSUPPORTED_REQUEST` before the source is read | No released Kumo Relational model serves candidate-free temporal link prediction, so the client refuses it up front, naming `RANK TOP k` with `candidates=` |

All 315 DuckDB cases were exercised through the clean installed native client during effort 2.
This is local scripted-endpoint coverage, not new live-model or performance acceptance.
`sqlite-python.json` covers 297 cases and `postgres-python.json` covers 313 cases, also checked through the installed client.
Their differences from DuckDB follow the fixture loader's explicit mappings in `contract/runners/databases.py` and existing backend behavior:

| Backend | Difference from DuckDB |
|---|---|
| SQLite | Decimals are stored as text, so numeric operations or explicit numerical type edits are refused; nested values and intervals become readable text features |
| SQLite / Postgres | Wide unsigned keys remain exact using text / numeric storage; nested map, struct and union fixtures are read in the backend's mapped representation |
| SQLite / Postgres | Malformed text-time values are refused; the US-date fixture completes under these backends' existing read behavior |
| Postgres | NaN-key fixtures refuse at materialize; a negative neighbor bound retains the existing `PROVIDER_ERROR` outcome |

These expectations describe the declared fixture mappings; they do not claim that every backend supports identical physical types.
`explain.two-entities` now succeeds on all three backends: The client supports explanations for multiple prediction rows.
`contract/tests/test_query_options.py::test_each_explained_entity_retains_its_own_raw_response` independently checks both rows' distinct explanation values, the request batch size and the public success outcome.
Rust/Java host expectations remain under reconciliation.
