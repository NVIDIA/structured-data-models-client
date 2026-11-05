<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Phase 10: simplify tests and remove migration baggage

**Status: plan for review; implementation has not started.**
Branch: `rust-core-phase-10`, based on `rust-core` at `ebc574639`.
The requested deletion of `docs/design/rust-core` is already included in that base.
Keep this plan only while cleanup is active; put lasting guidance in CONTRIBUTING and the developer change guide.

## Goal and limits

Keep tests that establish current product correctness, compatibility between Python and Java, and supported saved-data compatibility.
Remove old-client execution, migration administration, duplicated data and unnecessary test infrastructure.
No product/API changes, sampling changes, explanation fixes, license changes or Git history rewriting belong in this cleanup.
Old expectations are retained only when they express an intentional current contract, not merely because the old client produced them.

## Three steps

| Step | Changes | Evidence required before deleting |
|---|---|---|
| **1. Remove historical machinery** | Keep only current native protocol schema v71, current PQL schema and state-payload schema. Replace schema-history preservation checks with current generation/hash/version checks. Remove optional old-client execution, baseline-only tests, historical waivers, retired-route traps and obsolete recording/provenance checks. | Current Python/Java native version mismatch checks and schema/attachment validation still pass. Identify and retain helper functions imported by current tests before removing recorder scripts. |
| **2. Shrink fixtures without weakening assertions** | Generate the 100,000/100,001 entity inputs deterministically. Replace the 12.51 MB reference-observation bundle with minimal expected facts for its 11 consumers. Consolidate repeated shop datasets using one base plus simple explicit variants. | Same boundary outcomes, labels, values, ordering, nulls, errors and relevant payload fields remain asserted. Expected answers stay independent of the implementation. Both language runners consume the same expanded data. |
| **3. Organize maintained correctness checks** | Rename `tests/parity` to `tests/compatibility`; update CI, Java and docs together. Group retained contract tests by behavior rather than migration milestone. Consolidate overlapping harness helpers and tests. Simplify the existing capability manifest and remove obsolete approval history. | Preserve API-to-test links, applicable model/connector/type coverage, language exceptions and required test execution. Test collection and existing local/live entry points work under their new paths. |

Do these sequentially, in reviewable commits; avoid a simultaneous global rename and fixture rewrite.
Within a step, independent fixture families can be delegated once their consumers are identified.
Do not build a new fixture framework, ledger, recorder, reporting system or permanent deletion inventory.
Use ordinary test helpers and existing checks.

## What stays

- Apache-2.0 licensing and required third-party notices.
- Public API, wire-format, native ownership, sampling, SQL, error and result correctness tests.
- Current Python/Java consistency checks with independent expected outcomes; agreement alone does not establish correctness.
- Current connector journeys and CPU-model smoke checks, including required-case and skip detection.
- Historical serialized requests when needed to verify an explicitly supported saved-data format; generating them with today's client would invalidate that test.
- Raw Arrow/Parquet/timezone fixtures where exact encoding is the behavior under test.

The 203 binary fixtures total only 1.56 MiB and are all referenced; no blanket binary deletion is planned.
Reduce overlapping scenarios only when the same behavior remains covered.
Likewise, the roughly 127 infrastructure-oriented contract test functions are a review pool, not an automatic deletion list.

## Review rule for each removal

The commit description names the behavior being retained and its replacement test, or explains why the removed check only validated migration bookkeeping.
For consolidated tests, compare actual assertions and collected parameter cases, not just file names or total test counts.
Do not silently drop connectors, language paths, raw-reader cases, dtype/stype cases or unsupported-operation checks.
Keep current PR-target API compatibility enforcement; retire historical approval entries only after proving the target comparison no longer needs them.
Never hand-edit generated schemas or golden snapshots; update their source/generator and regenerate through maintained commands.

## Validation and completion

1. Before each step, record the affected test selection and existing fixture consumers; reuse recent valid results rather than rebuilding everything.
2. After it, run affected Rust/Python/Java tests and relevant schema, ledger or runner checks.
   When changing a runner, prove missing/failed/skipped required cases still fail its gate.
3. After integration, run maintained credential-free suites, lint/type checks, hygiene, schema/API checks and repository-size checks using warm builds.
4. Exercise changed connector-matrix entry points and bounded CPU-model smoke once at the end.
   If provisioning, comparison semantics or raw input handling changes, also run the affected hosted connector shards; lack of access leaves that validation pending.
5. Report before/after tracked bytes, text lines, fixture/schema files and test selections, plus what coverage was deliberately removed or replaced.

Nine obsolete schemas alone should remove about 109,000 lines / 3.15 MB.
The two giant input files offer another 6.16 MB reduction; shrinking observations offers more, but the retained facts determine the final size.
Do not meet a size or file-count target by weakening coverage, compressing everything into opaque blobs, or minifying JSON.
Completion means useful correctness checks and CI enforcement survive with substantially less data and machinery.
Merge/push remains a separate step after review; this plan does not authorize publishing.
