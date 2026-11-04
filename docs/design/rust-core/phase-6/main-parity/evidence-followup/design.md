<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Close the uncertain audit findings

**These are not confirmed bugs or approved new features.**
Resolve them with focused evidence before adding product code.

| Item | Old behavior / purpose | Proposed next step |
|---|---|---|
| Empty prediction result | Main's `d90d0a802` preserved whether the result had an ID column even when no rows were returned. This lets callers rely on the result schema. | Locate or add a public empty-result regression with/without IDs; fix shared Rust result layout only if a defect reproduces. |
| Repeated request planning | The same merge guarded against a second planning operation changing metadata IDs in a previously returned request. | Capture the first planned request, plan again, and assert the first remains unchanged. Fix ownership at the responsible layer only if it fails. |
| PostgreSQL timezone/null parameters | `3ea1936db` tested timestamp parameter precision, offsets and missing values. Rust has precision/null counterparts but the exact combined case was not mapped. | Test actual PostgreSQL-bound values with fractions, a timezone offset and null. Compare instants/precision, not offset spelling; UTC normalization is valid. |
| Engine diagnostic records | `2992c4b36` added structured engine prediction records and a SHA256 graph fingerprint. The old public client used materialization, not that instrumented prediction method. | Recommend explicitly retiring this engine-only utility unless a supported user workflow needs it. Do not equate Rust's sampler memo key with a diagnostic graph identity. |

The first three are regression-test contracts, not separate documented public methods.
No dedicated user notebook was identified for these exact edge cases, nor a public-client example relying on those engine diagnostic records.
Existing tests are evidence to reuse, not a requirement to revive the old engine test harness.

If diagnostics are later wanted, define the information and privacy contract first: Rust may supply meaningful execution facts, while the host emits logs.
Do not add hashing/logging machinery solely because the old engine had it.
A decision to retire the old utility should be recorded in the audit rather than silently called parity.

Acceptance: each regression is mapped to a maintained passing test or a reproduced defect with an explicit owner; diagnostic retirement is recorded separately.
No model calls are necessary for these ownership/schema/parameter checks.
