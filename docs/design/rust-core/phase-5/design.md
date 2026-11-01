<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Phase 5: standalone Rust-backed client

**Goal:** remove the old execution dependency, preserve public behavior, and prepare a verified release candidate.
Phase 4 is merged into `rust-core`; Phase 5 implementation has not started.

```text
Python / Java API
       ↓
Rust: validate query → prepare/sample data → build model request
       ↓
Python / Java: database and HTTP I/O
       ↓
Rust: interpret results → Python / Java: present them
```

Python/Java keep native data conversion, credentials and connections.
Rust keeps computation and policy, including both in-memory and warehouse push-down sampling.

## Three workstreams

| Work | What to do | Done when |
|---|---|---|
| 1. Remove legacy execution | Extract still-needed host helpers, remove routing overrides and old computation, then remove obsolete dependencies | Clean-installed clients work without the old engine/parser distributions; API and supported pickle behavior remain intact |
| 2. Close verification | Audit existing performance, runtime, contract and live results; reuse applicable evidence and rerun affected or missing checks | Required checks have passing evidence or an explicit owner-approved disposition |
| 3. Prepare release | Reconcile current `main`; update packaging, extras, docs, notices and protected workflows; build clean wheels/JARs | Final artifacts pass required checks and are ready for the owner's merge/release decision |

Preserve the frozen reference, public regressions, PQL conformance and permanent `tests/parity` infrastructure.
Legacy modules still contain required host helpers, so deleting whole directories first is unsafe.

## Package boundaries

All names below have the `structured-data-models-` prefix.

| Package | Plan |
|---|---|
| `client`, `api`, `connectors` | Keep the public client, model wire types and source access |
| `pql`, `relational-engine` | Stop requiring their old implementations in the new client after compatibility checks; their features already have Rust implementations |

**Removing a client dependency does not mean deleting historical PyPI releases.**
Whether to stop publishing either old distribution is a separate decision after reviewing standalone consumers and compatibility needs.
Preserve supported install extras and any necessary data-only pickle compatibility shims.

## Optional external PQL integration

Source review at `04cfe114` found two distinct components in the separate PQL repository:

| Component | Possible replacement here |
|---|---|
| Language crate: parsing and schema-aware validation | Our Rust `sdm-pql` |
| Engine crate: labels, task preparation and in-memory neighborhood sampling | Parts of our local materialization/sampling implementation |

Its current context builder uses a local sampler; no warehouse push-down implementation was found.
Adopting only its language crate would leave both sampling paths here; adopting its engine would need a separate compatibility review.
Neither automatically replaces connectors, model transport, sessions or Tabular encoding.
Keep our current Rust implementation for Phase 5; external adoption is optional, not a cleanup prerequisite.
If adopted later, integrate through Rust so both languages share it.

## Verification and status

Phase 3 already has substantial verification and approved performance costs; Phase 4 adds 792 comparisons across identified builds.
The [verification tracker](verification.md) records what can be reused and what still needs checking.
Preserve original build identities; unaffected warehouse/model reuse needs two independent reviews and final green full CI.
New packaging still needs clean-install verification; dedicated live-CI infrastructure remains deferred.

All three workstreams are **planned**, with initial dependency/evidence investigation complete.
No new model functionality or sampling redesign is required.
Merging into `main` and publishing require owner approval.
[Disk cleanup](cleanup-result.md) is separate maintenance; its completed batches do not establish release readiness.
