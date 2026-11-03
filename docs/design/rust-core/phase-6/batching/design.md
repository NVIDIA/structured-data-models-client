<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Batch prediction and automatic context reuse

Approved follow-up to [remaining API mappings](../remaining-api-mappings.md).

```text
Python / Java: tables + rows_per_call → Rust materializes batches
Rust: identical context/task/schema/output/inference?
  Yes → create temporary session → score batches → delete
  No  → ordinary prediction calls
Python / Java: HTTP and bounded backoff only
Rust: validate answers → return combined results or completed partial results
```

| Contract | Rule |
|---|---|
| Batch size | Supplied examples gain Python `rows_per_call` and Java `ExamplesOptions.rowsPerCall`; Rust caps positive values at 10,000 rows, 200 for temporal links and one for forecasting. |
| Reuse | Ordinary multi-call Relational requests only, transport supports sessions, identical actual wire context/task/schema/output/inference and metadata except request ID; no cache across requests. |
| Different contexts | Existing stateless calls; fixed random seeds alone prove nothing. |
| Expiry | One recreation per prediction operation after scoring returns 404; previously completed batches are never repeated. |
| Unsupported create | HTTP 404/405/501 selects stateless fallback. Other failures propagate. |
| Retries | One host transport budget: 408/429/502/503/504 plus existing connection failures; no outer batch retry loop and no retries of uncertain session creation. HTTP500 remains excluded. |
| Cleanup | Attempt deletion of every known session ID on success or failure; cleanup failure warns without replacing results or primary errors. |
| Explicit fit | Existing API and lifecycle unchanged. |

Tests use small public API requests through real local HTTP servers in both languages.
Cover identical/different contexts, batch limits, creation/fallback/expiry, bounded retries, cleanup, partial results and unchanged explicit fit.
Use the existing authorized session endpoint for one synthetic live lifecycle; no new deployment.
Compared with the old client, this prevents seed-based unsafe reuse and retry multiplication while retaining completed answers.

The supplied-example path previously put 201 temporal-link rows in one payload despite the existing 200-row model limit.
A public reproduction on contract 65 confirmed this; the restored shared limit produces batches of 200 and one.

## Verified on production contract 67

| Check | Result |
|---|---|
| Rust contracts, prediction, multicall, serving and session tests | 92 passed; Clippy clean. |
| Python HTTP lifecycle and affected existing suites | 129 passed; 10 optional tests skipped in the local run. |
| Java HTTP lifecycle and affected existing suites | 44 passed; the live test was run separately. |
| Existing model endpoint | Python and Java both passed with the same small shared dataset. Python observed one creation, three scoring calls and one deletion, then compared the identical prepared request through stateless calls: row IDs and order match, predictions match within `rtol=1e-5`, `atol=1e-6`. |
| Boundary checks | Wire snapshots, contract attachment/schema locks, Ruff and targeted mypy passed. |

The two live workflows ran separately from the local suites; no deployment was created.
Explicit sessions retain their existing behavior, including failed-close retry and session-ID aliases.
