<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Maintained observations

These fixtures hold independent expected facts for current correctness tests.
Tests need only this checkout; historical clients, archives and recording tools are not dependencies.

- `request-expectations.jsonl` holds exact small request payloads, schemas, results, warnings and success/refusal outcomes.
  It omits historical checkout hashes, recorder identities and unused protocol envelopes.
- `batch-expectations.json` holds literal sampled entity order and request options for the two large, single-table batching cases.
  The test reconstructs every expected request from that independent order and the input table's literal `score = entity % 7` rule.
  Each call excludes its scored entities from context while retaining sample positions, so full request comparisons still check ordering, labels and every row.
- `label-expectations.json` contains only the four label cases' expected context rows and counts.
- `duckdb-statements.json` contains the expected SQL statements and their results for the provider regression case.
- `postgres-statements.json` retains every distinct PostgreSQL statement with its original case and multiplicity, plus all observed key-lookup column/type layouts.
  Write-guard and provider-layout tests exercise these facts without loading full driver result logs.
- `snowflake.json` contains the statements and result references for the 27 Snowflake cases named by `snowflake.rs` and the public Python Snowflake tests.
  `snowflake/` shares identical Arrow IPC streams by their SHA-256 names.
  Original driver Arrow bytes are preserved, including their types and metadata.
- `snowflake-public.json` shares the independent public schema and wire-payload expectations for the 14 cases named by the Python Snowflake consumer.
  Both consumers exercise these expectations.
- `decimal-key-inputs.json` contains the setup and indices used by the current exact-decimal join tests.
  Those tests assert exact joins against ordinary-key twins and literal entity identities; they did not consume the historical decimal predictions.
- `consumers.json` names the current consumers and reviewed byte limits for large retained fixtures.
  New oversized fixtures require an explicit entry and explanation.

Eight fixtures elsewhere under `contract/` use `sdm-shared-fixture/1`.
Named fields have dictionaries of literal values; ordered rows reference those dictionaries, with `-1` for an absent field.
Nested tables can reference only earlier tables.
This shares repeated inputs and expected results while preserving every case, its order, nulls and JSON types.
The expanded SHA-256 binds the complete original logical JSON document.
The Python and Rust test helpers expand these literals without calling product code to compute expectations.

The original extraction tool and recording-completeness tests were migration-only and have been removed.
Git history retains their implementation and original fixtures.
Maintain expected values from documented behavior and independently calculated examples; do not regenerate expected answers from the product under test.
Ordinary tests need only this checkout.

The retired `column-name-vectors.json`, `key-cast-vectors.json` and `nested-type-vectors.json` had no current fixture readers.
Current column-name tests generate their inputs from the maintained case builder and retain their independent behavior assertions.
The decimal observation archive was replaced by the consumed inputs described above.
Large repetitive batching inputs use documented deterministic case generators.
Current-run input identity checks remain in the runners; stale historical input-file hashes are not expected behavior.
