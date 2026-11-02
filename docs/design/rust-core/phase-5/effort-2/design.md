<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Effort 2: one runtime, a maintainable repository

Remove the old implementation and migration-only machinery while preserving supported behavior and useful regression coverage.
This follows the [Phase 5 plan](../design.md) and the [compatibility contract](../effort-1/design.md).
No merge into `main` or release is included.

## Intended result

```text
Python / Java public APIs
          ↓
Thin host adapters + shared Rust core
          ↓
Connectors and model endpoints

Permanent tests → explicit expectations + compatibility ledger
Archived old-client comparisons → available for effort 3 investigations
```

Keep the existing host/core responsibilities and warehouse push-down algorithms.
Removing Python legacy code must not move shared computation back into host adapters.

| Material | Action |
|---|---|
| Old execution, route switches and runtime dependencies | Remove after tracing callers and extracting genuinely needed host utilities |
| Migration orchestration, temporary scripts, captures and intermediate plans | Preserve a recoverable snapshot outside the maintained tree, then remove |
| Frozen client, comparison runners and original acceptance evidence | Keep available outside the maintained tree through effort 3; preserve provenance and a runnable comparison recipe |
| Useful regression cases, vectors and expected payloads | Retain as self-contained tests; ordinary CI must not install the old client |
| Compatibility ledger, build tooling and architectural rules | Retain and simplify; remove phase-specific naming and assumptions where appropriate |
| READMEs, API/install/contributor docs and examples | Describe the current architecture and supported workflows |
| Licenses and attribution | Preserve; regenerate generated notices when their inputs change |

The matrix is a coverage contract, not a replacement oracle.
Judge historical tests against the current public contracts, not one-for-one parity with old implementation details.
Delete obsolete helper, routing and duplicate tests when they add no missing product coverage.
Keep or replace tests that protect meaningful supported behavior; investigate differences as possible old or new bugs rather than automatically copying the old result.
Treat the Phase 4 Rust implementation as the behavioral baseline; change it only for a demonstrated product bug, not to satisfy obsolete tests.
Historical comparisons remain recoverable outside the maintained tree through effort 3.

## Revised plan: delete migration tooling, keep contract coverage

This plan supersedes the attempt to replace the historical comparison harness in full.
Deletion is the default for migration-only tooling; preserving useful coverage is the requirement.
Do not build a new migration framework or carry every historical case forward.

1. **Decide what protects the product.** Identify the current contract each remaining test or tool enforces and whether maintained tests already cover it.
   Keep unique, valid coverage; archive obsolete checks, duplicates and tests of the migration process itself.
2. **Separate the minimum, then delete.** Extract only the fixture, recording or replay helpers retained tests actually need.
   Remove historical comparators, approval registries, route-switch checks and orchestration from normal development and CI after verifying recoverable archives.
   Update their callers and documentation together.
3. **Validate the retained system and finish.** Review boundaries and coverage before the final test pass; run the relevant Python, Rust, Java, packaging and CI checks after cleanup.
   Reuse valid existing evidence for unchanged areas, investigate real failures, and obtain independent review before a signed effort-2 commit.

| Keep when it adds useful coverage | Archive/remove |
|---|---|
| Public API and behavior tests; compatibility ledger; architecture checks | Tests of old internals, routing transitions and migration approvals |
| SQL goldens and provider tests that protect warehouse push-down behavior | Historical deviation handling and checkpoint/ratchet machinery |
| Replay fixtures or cross-language checks for the same operation and inputs | Whole-trace equality between runners exercising different workflows |
| Small shared fixture/observation helpers needed by retained tests | Recorders, captures and scripts with no ongoing test/build purpose |

A Python end-to-end workflow and a Rust parser-only test may both be valuable without producing identical traces.
Give each explicit expectations for its own contract; compare languages directly only where operation, inputs and scope match.
Do not change product behavior or weaken assertions merely to make unlike runners agree.

Keep completed cleanup and useful Python CI improvements.
Reassess the new observation checker and expectation files under the same rules as older tooling; work already spent is not a reason to retain them or extend them to every historical runner.
Record intentional coverage changes in the existing effort-1 ledger where applicable, rather than creating another tracking framework.

The completion note should briefly state what was kept, deleted and archived, where coverage lives, and what remains for effort 3.
There must be no normal test/build/CI dependency on the external archive or historical approval machinery.

## Compatibility decisions

Existing pickle behavior is part of the contract.
Investigate which old names are still needed; if strict old-name removal conflicts with compatibility, present a concrete choice before breaking it.
Removing dependencies is separate from retiring published packages or standalone consumers.
Do not delete historical PyPI releases or assume the old engine has no independent users.
The owner approved archiving the standalone Python PQL and relational-engine packages and stopping their builds once maintained-runtime independence is verified.
Their original trees are recoverable in the verified external archive; dependency inspection and clean installed-client checks confirm neither package is required by the maintained runtime.
This does not establish that external users of historical standalone distributions no longer exist.
External PQL adoption and new features are outside this effort.

## Completion

Normal installation, execution and CI use only the maintained implementation and tests.
Every retained file has an ongoing purpose; migration-only material is recoverable outside the maintained tree.
Required coverage remains enforced, documentation matches the code, and independent review and relevant checks pass.
Effort 3 owns final runtime/performance acceptance, reconciliation with `main`, and the merge decision.

## Completion record

Implementation, local validation and independent review are complete.
The [completion summary](completion.md) records what was kept, removed and archived, final checks and effort-3 obligations.
This plan and its completion record are included in the signed-off effort-2 commit on `rust-core-phase-5`.
No merge into `main` or release was performed.

Verified snapshots, checksums, restoration recipes, reviews and detailed logs are outside the repository at `~/.local/share/sdm-orchestration/phase-5/effort-2/`.
The archive's `README.md` and `original-comparison-recipe.md` describe restoring frozen clients for effort 3.
`final-evidence.json` records the final local checks; `progress-before-final-summary.md` preserves the earlier detailed cleanup history.
The maintained tests and CI require neither archived clients nor machine-specific migration tooling.

The acceptance-evidence limits in [verification.md](../verification.md) remain unchanged.
