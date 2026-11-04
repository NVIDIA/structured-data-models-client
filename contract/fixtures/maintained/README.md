<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Maintained observations

These fixtures retain facts consumed by current tests from the migration recordings at commit `4e3f4a86f`.
The full recordings are an external archive, not a dependency of tests or CI.
`PROVENANCE.json` records the archive, source-file and extraction-tool SHA-256 digests.
The archive's external manifest additionally records every original file, its byte length and digest.

- `reference-observations.jsonl` contains the 345 observation cases named by the Python and Rust consumers in the provenance file.
  Driver recordings are omitted except for the three complete observations used by SQL-result and observation-schema tests.
  Public payloads, result/schema expectations, outcomes and identity hashes are unchanged.
- `reference-inventory.json` preserves every original case identity and captured-statement count for the recording completeness checks.
- `postgres-statements.json` retains every distinct PostgreSQL statement with its original case and multiplicity, plus all observed key-lookup column/type layouts.
  Write-guard and provider-layout tests exercise these facts without loading full driver result logs.
- `snowflake.json` contains the statements and result references for the 27 Snowflake cases named by `t36_snowflake.rs` and the public Python Snowflake tests.
  `snowflake/` shares identical Arrow IPC streams by their SHA-256 names.
  Original driver Arrow bytes are preserved, including their types and metadata.
- `snowflake-public.json` shares the independent public schema and wire-payload expectations for the 14 cases named by the Python Snowflake consumer.
  The extractor reads both consumers when selecting cases, so Python-only cases are not lost when removing capture archives.
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

Regenerate and independently verify the extraction with the archived originals:

```bash
python scripts/rust_core/extract_fixtures.py --archive /path/to/original-contract-schema.tar
python scripts/rust_core/extract_fixtures.py --archive /path/to/original-contract-schema.tar --verify
python scripts/rust_core/repository_size.py --base 847a1e9e3
```

The extractor requires the test environment because executable test declarations supply the exact reference-case inventory.
It never runs tests, the current engine, or an old client to produce expected answers.
Verification compares every expanded vector, projected observation and retained Arrow stream against the archive.
Ordinary tests need only this checkout.

The retired `column-name-vectors.json`, `key-cast-vectors.json` and `nested-type-vectors.json` had no current fixture readers.
Current column-name tests generate their inputs from the maintained case builder and retain their independent behavior assertions.
The decimal observation archive was replaced by the consumed inputs described above.
The large batching cases remain unchanged because their byte-level input identities are used across host-language contracts.
The repository gate bounds their growth explicitly.
