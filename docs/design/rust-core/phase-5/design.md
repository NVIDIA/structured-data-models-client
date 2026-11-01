<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Phase 5: make `rust-core` ready for `main`

**Proposal for owner review, not the final implementation plan.**
Deliver one implementation, clear API/architecture docs, and maintained compatibility checks.
Implementation waits for review; merging/publishing require owner approval.

## 1. One implementation, clearly documented

```text
Public Python / Java APIs
          ↓
PyO3 / JNI → Rust host boundary → shared Rust computation
                                      ↕
                          host database / HTTP I/O
```

| Boundary | Proposed responsibility |
|---|---|
| Python/Java client and connectors | User objects, conversion, credentials, connections, HTTP and presentation |
| `sdm-host` / `sdm-contract` | Native/Arrow admission and the typed host/core protocol; distinct from user-facing APIs |
| `sdm-core` | Compose operations and coordinate host requests |
| Domain crates | `sdm-pql`/`sdm-schema`: query/schema rules; `sdm-engine`/`sdm-data`: sampling and SQL; `sdm-request`/`sdm-wire`: model requests/results |

Define Python/Java APIs, errors, ownership, persistence and the supported Rust embedding surface in `docs/reference/stability.md`.
Internal modules are not automatically public APIs.
Extract needed host utilities, then remove old execution, routing and dependencies.
Archive migration-only code/docs/oracle outside the maintained tree; permanent tests must not require the old client.
Update root/Rust/Java READMEs, API/install/contributor docs, examples and packaging.
Retain legal attribution and historical Git evidence, not obsolete product guidance.

## 2. One compatibility contract, enforced in CI

Extend **`tests/parity/manifest.json`**, not a second ledger or testing framework.

| Contract | Proposal |
|---|---|
| Coverage dimensions | Language × connector/input path × model × capability: task, lifecycle/API, outputs and semantic/physical type families |
| Tasks/types | Node regression/classification, temporal/link/ranking/forecast, `stype`/`dtype`, nulls and conversion boundaries |
| Applicability | Declare supported, refused or not-applicable with a reason; equivalent capabilities across languages, with idiomatic APIs and explicit model/connector limits |
| API changes | Every added/changed/removed Python or Java API requires capability mapping and executable tests in applicable languages, or an approved exception |
| Missing coverage | Generate required cases from capability declarations, then check scenario coverage; removal needs a recorded decision, not a silently disappearing row |
| Readable matrix | Generate from contract/results; distinguish promised support from tested coverage and identify artifacts |

Current gaps: Java surface checking, complete API mapping and explicit type inventory; omitted scenarios can hide capabilities.
Test type/refusal rules thoroughly in Rust and conversions at host/connector boundaries; representative integration cases do not prove every combination.

| CI tier | Proposed checks |
|---|---|
| Every PR, no GPU | API/manifest coverage, core behavior/types/refusals, host conversion, local connector and protocol tests; shard existing Rust/pytest/JUnit runners |
| Protected manual/scheduled/release | Real warehouses/model endpoints and required platform/performance checks; isolated fixtures, bounded endpoint concurrency, giant-memory cases sequential |

Require all expected case/shard results; missing/skipped/duplicate/inconsistent evidence fails.
Build artifacts once, reuse across shards and measure duration before choosing a CI budget.
Reuse applicable [existing evidence and approved costs](verification.md); final-source/clean-install checks remain required.
Dedicated live-CI credentials are deferred; local live validation remains available.

## 3. Reuse tools before building more

| Proposal | Why |
|---|---|
| Existing Python snapshots, manifest and fixtures | Extend coverage checks; replace routine old-client comparisons with maintained behavior expectations |
| Add [japicmp](https://siom79.github.io/japicmp/MavenPlugin.html) | Java API diffs; capability mappings/tests still prove behavior |
| Reuse [pytest-xdist](https://pytest-xdist.readthedocs.io/en/latest/distribution.html), [JUnit parallelism/resource locks](https://docs.junit.org/5.13.1/user-guide/index.html) and CI matrix jobs | Parallelize existing tests without a custom scheduler |
| Consider [cargo-semver-checks](https://github.com/obi1kenobi/cargo-semver-checks) | Only for the declared stable Rust facade, not internal crates |

No reviewed tool enforces the whole behavioral matrix.
[UniFFI](https://mozilla.github.io/uniffi-rs/latest/) generates bindings, lacks first-party Java support, and does not solve parity; keep PyO3/JNI.
External PQL adoption stays optional: language and local-engine replacements are separate; warehouse push-down remains ours.

## Review decisions and completion

- Approve one capability ledger with equivalent Python/Java functionality, allowing explicit applicability differences.
- Decide old-pickle compatibility: an explicit exception for data-only old-name shims, or a separate migration converter; strict removal cannot silently retain old module references.
- Decide package retirement separately from dependency removal; do not delete historical PyPI releases.

**Done:** new runtime only, current docs/API boundaries, enforced compatibility contract, required acceptance evidence and clean final artifacts, with current `main` reconciled and no unexplained failures or missing required coverage.
Details stay in the verification tracker; [disk cleanup](cleanup-result.md) is separate.
