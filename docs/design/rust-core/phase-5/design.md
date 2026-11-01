<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Phase 5: retire the old runtime and prepare release

**Three workstreams; no new sampling architecture.**
Start from Phase 4 merged at `2b9e6ea88`.
This is a plan, not authorization to delete worktrees, merge into `main` or publish.

```text
Today:  Python API → route switch → Rust or old Python computation
        Java API ────────────────→ Rust computation

After:  Python / Java API → Rust policy → host I/O → model
                                      ← Rust result handling
        Frozen old Python remains isolated for comparison tests.
```

## 1. Remove the old execution path

Rust retains schema/PQL rules, sampling, request construction, session policy and result attribution.
Python/Java retain native conversion, credentials, connections, HTTP and presentation.

| Change | What must survive |
|---|---|
| Remove `_routing.py`, `SDM_CORE_ROUTES` and old computation branches | Public behavior and supported model/type limits |
| Extract host helpers before deleting legacy modules | Native code still imports Tabular input helpers, request IDs/error presentation, limits and the `EngineTables` holder from those modules |
| Remove obsolete runtime dependencies from the new client | Replace use of the old engine/Python parser with shared Rust; preserve extras/import/pickle compatibility where required. Retiring their separately published distributions is a distinct release decision |
| Retire migration-only tooling selectively | Keep public regressions, PQL conformance, frozen reference and permanent `tests/parity`; move required build/release tools first |

**Exit:** clean-installed clients work without `sdm_relational_engine` or `predictive_query`; old routing is unavailable and public/persistence regressions pass.
This does not delete historical PyPI releases or remove relational/PQL features.
Standalone consumers of the old distributions and compatibility shims must be reviewed before deciding whether to stop publishing those distributions.
The current implementation keeps Rust `sdm-pql`; possible adoption of a separate language/engine library is a future integration decision, not a prerequisite for Phase 5 cleanup.

## 2. Close verification by auditing first

The [verification tracker](verification.md) records the existing receipts and accepted costs.
Phase 3 already completed substantial formal, runtime and performance testing; Phase 4 added 792 comparisons across identified builds.

| Area | Next action |
|---|---|
| Performance | Apply existing measured-cost approvals; benchmark changed paths and genuine gaps, not accepted costs all over again |
| Runtime/platform coverage | Map earlier runtime results to changed bindings/dependencies; test missing or affected cells and final packaging |
| Formal regression gates | Reconcile contract/differential/held-out receipts and approved differences; rerun affected or missing checks |
| Live parity | Reuse frozen observations and proven-unaffected cases; exercise routes changed by engine removal |

Record **requirement → original receipt/build → source/dependency changes → reuse or rerun → review**.
Existing reuse approval requires two independent reviewers for unaffected warehouse/model results and final green full CI.
Keep original build IDs and raw failures with their approved dispositions.
Dedicated live-CI infrastructure remains deferred; local live evidence can still be used.
Documentation alone cannot prove a newly packaged, engine-free artifact works.

## 3. Prepare one release candidate

Reconcile current `main` before freezing the candidate.
Update extras, install docs, support declarations, attribution/SBOM and build/publish workflows together.
The protected live workflow currently allows only `main`/`rust-core-phase-4`; deliberately update its Phase 5 policy without exposing credentials to arbitrary branches.
Build clean wheels/JARs and complete the remaining installed-artifact/platform acceptance.
Owner approval remains required for merging into `main` and publishing.

## Tracker

| Workstream | Status | Completion condition |
|---|---|---|
| Runtime cleanup | Planned; dependency traps identified | Engine-free installed clients with preserved API/pickle behavior |
| Verification | Existing evidence audited; applicability mapping next | Every required check proved, explicitly disposed by the owner, or clearly left open |
| Release candidate | Not started | Final artifacts/checks and owner decision packet ready |

This is narrower than Phase 4, but not just deleting a directory.
Legacy helper imports, pickle paths and final packaging are the main uncertainties; no new model feature is currently required.
[Repository cleanup](cleanup-audit.md) is separate maintenance, not another implementation package.
