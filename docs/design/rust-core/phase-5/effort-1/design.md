<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Effort 1: define and enforce the compatibility contract

Make changes to the supported Python/Java APIs and capabilities visible in review, with fast checks that catch missing coverage.
This implements the first step of the [Phase 5 plan](../design.md).
It does not remove legacy code, change sampling, add model features, or merge into `main`.

## What belongs where

```text
Python / Java public API
    | user objects, conversion, credentials and I/O
    v
PyO3 / JNI → sdm-host + sdm-contract
    | typed operations and Arrow data
    v
sdm-core → domain crates
          query/schema rules, sampling, SQL, model payloads/results
```

Document the supported user APIs in `docs/reference/stability.md`.
Host adapters perform I/O; Rust owns shared computation and validation.
The native protocol is an internal interface, not another user API.
Do not promise stability for every public symbol in internal Rust crates.

## One ledger, two kinds of evidence

Extend `tests/parity/manifest.json`; keep the existing runners and fixtures.

| Record | Required information |
|---|---|
| API | Public Python/Java symbol, capability, executable tests or a specific applicability exception |
| Capability | Stable ID; applicable languages, models and inputs; supported/refused/not-applicable with reasons |
| Types | Semantic/physical type families and null/conversion cases, with their existing tests |
| Coverage | Required cases derived from capability declarations, separately from scenario selection |
| Changes | Explicit explanation for retirement or changed applicability; missing mappings fail validation |

A passing local test proves its stated boundary, not a live model prediction.
Keep existing live receipts and distinguish support promises, local coverage and endpoint evidence.
Do not expand every type into every live connector combination.

## Three implementation steps

1. Inventory the actual public surfaces and existing tests; add complete capability/type mappings and clarify API ownership.
2. Enforce API and coverage changes: retain generated Python snapshots, add Java public API comparison using japicmp, and reject unmapped APIs, missing scenarios and silent retirements.
3. Wire credential-free checks into PR CI using existing pytest/JUnit/Cargo runners; verify expected test identities and reject missing, skipped or duplicate required results.

Use a pinned, reproducible Java baseline and exclude internal packages from public API comparison.
Compare ledger changes against the target branch merge base so deleting both a capability and its tests cannot hide a retirement.
Check every current public Java symbol as well as compatibility differences, including additions.
Reconcile selected test identities, including parameterized cases, against native runner reports.
API comparison detects surface changes; the ledger and behavior tests establish cross-language capability coverage.
Build once where practical and parallelize independent tests with existing tooling.
Keep live endpoints, warehouses and giant-memory checks outside this fast gate.

## Done when

- Public APIs and ownership are documented and all declared APIs/capabilities have mappings or explicit exceptions.
- Deliberately adding an unmapped API, deleting required coverage, or omitting a required result makes the gate fail.
- Existing relevant behavior tests pass, with measured local gate duration and independent review of the contract and implementation.
- Remaining live/performance acceptance stays explicitly tracked in [verification.md](../verification.md), without relabeling old evidence as a new run.

## Progress

Implementation and independent review are complete; results and evidence limits are in [completion.md](completion.md).
Use the [maintained commands](../../../../../tests/parity/README.md) to check API changes and required test execution.
