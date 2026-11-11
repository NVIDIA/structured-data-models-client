<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Maintained observations

These fixtures hold independent expected facts for current correctness tests.
Tests need only this checkout; historical clients, archives and recording tools are not dependencies.

- `snowflake.json` contains the statements and result references for the 14 cases used by the public Python Snowflake tests.
  `snowflake/` shares identical Arrow IPC streams by their SHA-256 names.
  Original driver Arrow bytes are preserved, including their types and metadata.
  Rust tests use independent assertions and four small original Arrow conversion samples directly.
- `decimal-key-inputs.json` contains the setup and indices used by the current exact-decimal join tests.
  Those tests assert exact joins against ordinary-key twins and literal entity identities; they did not consume the historical decimal predictions.
- `consumers.json` names the current consumers and reviewed byte limits for large retained fixtures.
  New oversized fixtures require an explicit entry and explanation.

The original extraction tool and recording-completeness tests were migration-only and have been removed.
Git history retains their implementation and original fixtures.
Maintain expected values from documented behavior and independently calculated examples; do not regenerate expected answers from the product under test.
Ordinary tests need only this checkout.

The retired `column-name-vectors.json`, `key-cast-vectors.json` and `nested-type-vectors.json` had no current fixture readers.
Current column-name tests generate their inputs from the maintained case builder and retain their independent behavior assertions.
The decimal observation archive was replaced by the consumed inputs described above.
Large repetitive batching inputs use documented deterministic case generators.
Current-run input identity checks remain in the runners; stale historical input-file hashes are not expected behavior.
