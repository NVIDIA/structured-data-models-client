<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Remove migration bulk before merging to main

**The 22 million added lines are real: retained migration recordings dominate, not Rust code.**
Audit: `rust-core` after Phase 6, `db8f88528`, against main `847a1e9e3`.
The audit below records the pre-cleanup state; Phase 8 implementation and verification are recorded at the end.

## Size and cause

| Measurement | Finding |
|---|---:|
| GitHub-style diff from merge base (`main...rust-core`) | **22,564,336 added / 93,552 deleted lines** |
| Current tree versus main (`main rust-core`) | 22,564,133 added / 97,886 deleted lines |
| Entire tracked tree | **1.651 GB; 19,879 files** |
| `contract/captures/` | **20,499,025 added lines; 1.509 GB; 17,900 files** |
| Capture contents | 484 MB JSON + 1,025 MB Arrow |
| Local shared Git objects | 335.65 MiB loose + 180.10 MiB packed, across all local branches/history |

Captures account for **90.85% of added lines**.
Four recommendation captures alone contain approximately 1.6 million lines each.
The diff totals rule out a display artifact.
The branch ported main features; it was not rebased onto latest main.

```text
Old-client warehouse runs
    → complete SQL/results and prediction observations
    → duplicated per-case/provider capture files
    → tooling removed, bulk recordings retained
```

Phase 3 commit `6c5c61d48` imported **9,311 files / 16,169,904 lines**.
Phase 5 commit `af4c801a9` deleted 5.5 million lines but left `contract/captures/` unchanged, contrary to its [minimum-fixture extraction plan](../../phase-5/effort-2/design.md).
Captures repeat driver logs, statement logs and full prediction payloads; Arrow deduplication happens only within each case.
Identical blobs occupy 353 MB of repeated checkout content.
Separately, `contract_lock.py --prune-working` removes lock entries but leaves old schema files behind.
There is no aggregate fixture-size or current-consumer gate.

## The other two million lines also need cleanup

After captures, **1.681 million remaining added lines are JSON/JSONL**.

| Remaining material | Added lines | Action |
|---|---:|---|
| Contract JSON cases/vectors/reference | 827,261 | Extract independent expectations; parameterize repetition |
| Versioned schemas | 617,975 | Remove 36 unlocked historical versions: 525,715 lines / 15.12 MB |
| Rust tests/testdata/manifests | 225,220 | Separate useful tests from large recorded corpora |
| Python/Java tests/fixtures | 162,027 | Keep unique product coverage |
| Rust/Python/Java source directories | 152,986 | Includes inline tests/testkit, not pure implementation SLOC |
| Other tooling, docs and goldens | 79,842 | Audit purpose |

For example, view-expression vectors record 13,333 calls; view-step vectors record 12,797 default-text results.
Whitespace compression alone does not address that duplication.

## Proposed change

1. **Extract the minimum, then remove archives.**
   Preserve full migration evidence outside the maintained checkout with hashes and provenance.
   Replace large recordings with the exact driver rows, SQL and expected results required by current tests.
   Extract used facts from the 60.19 MB reference JSONL instead of loading complete old observations.
2. **Reduce repetitive fixtures and stale schemas.**
   Share common inputs; generate large stress data at runtime; retain independent expected outcomes and unique regressions.
   Keep current/explicitly supported locked schemas and embedded `state-payload.json`; remove obsolete unlocked versions through maintenance tooling.
3. **Enforce the result.**
   Check tracked bytes/lines, oversized new fixtures, orphan schemas and named consumers in CI.
   Reuse Git metadata, not a new framework.

Deletion requires a consumer audit: `sdm-testkit/tests/t36_snowflake.rs` still reads captures.
Its literal case names identify 21 directories totaling 6.97 MB; trace dynamic reads before claiming that inventory is complete.
Other SQL tests already embed extracted statements, while `sdm-core/tests/view_captures.rs` reads a separate compact fixture.
Several `contract/tests` modules still read the reference JSONL.
Update those consumers and generators together; never hand-edit generated files or derive expected answers from the implementation being tested.

## Acceptance and Git history

Proposed budget: **under 100 MB tracked files and under one million added text lines** against this main baseline, with reviewed exceptions.
The remainder is **67.13 MB** after excluding captures, reference JSONL and obsolete schemas; the byte target leaves room for extracted regressions.
The line target requires reducing vectors too; neither target is achieved yet.
Require relevant Python/Rust/Java tests to pass offline without the archive and publish a before/after inventory.

Tree deletion fixes the PR diff but does not erase historical Git objects.
A reviewed squash/clean integration can avoid importing migration ancestry into main; old remote branches may still inflate ordinary clones until separately archived and retired.
History rewriting is a separate owner decision, not part of this cleanup.
LFS, minification and GitHub hiding rules do not remove unnecessary data.

Reproduce with `git diff --numstat origin/main...db8f88528`, `git diff --numstat origin/main db8f88528`, `git ls-tree -rl db8f88528`, and `git count-objects -vH`.

## Phase 8 execution plan

Preserve the original contract fixtures and schema directory outside Git in a checksummed archive before deletion.
Extract only consumed Snowflake statements/results and reference observations, with source hashes and extractor provenance.
Represent repetitive vector inputs and independent expected values as shared tables, preserving all expanded cases exactly.
Remove unlocked historical schemas through maintenance tooling, retaining all locked versions including 70 and the embedded state payload.
Add aggregate tracked-byte/added-line, fixture-consumer and orphan-schema gates, then run focused extraction and consumer checks.
The integration owner runs final correctness and performance checks after all four efforts merge.

## Implemented checkout cleanup

Phase 8 preserved 18,525 original contract/schema files in an external archive and per-file SHA-256 manifest before removing migration bulk.
The uncompressed archive SHA-256 is `7756758481b8e8af1be5fbb97c5f9b78b5e4e805411156d75e7f2e73e9fdea72`.
The compressed archival copy SHA-256 is `6f1a65c775d6f543964780420d55806876da6b050eb77fba6ec59ed938fc0ee3`.
The archive remains outside Git; retained fixtures carry source and extraction-tool provenance.

| Measurement | Before | After extraction |
|---|---:|---:|
| Tracked checkout bytes | 1,652,168,033 | approximately 75.3 million |
| Tracked files | 19,894 | approximately 2,130 |
| Added lines against main `847a1e9e3` | more than 22 million | approximately 978,000 |

The exact final integrated totals come from `scripts/rust_core/repository_size.py --base 847a1e9e3`.
The enforced limits are strictly below 100,000,000 bytes and 1,000,000 added lines.
The gate also rejects returning migration archives, unlocked versioned schemas, oversized fixtures without named consumers, missing consumers and corrupt shared-table expansions.
Generated API snapshots and the parity manifest retain their existing dedicated gates.

The extraction retains 345 consumed reference observations, 27 Snowflake statement/result cases and all cases in eight shared-table fixtures.
Fourteen public Snowflake schema/payload expectations are retained independently, including Python-only cases.
Every shared-table expansion is compared to its original independent recording, including order and JSON types.
Three orphan vector recordings were removed; decimal recordings were projected to the inputs used by current exact-join assertions.
The type-set and semantic-type fixtures and their parity hashes were preserved.
The two large batching inputs remain explicit to preserve their cross-language byte-level identities; they have reviewed byte bounds.
All nine locked contract versions, including newly merged version 70, remain unchanged, alongside validated-query version 2 and the embedded state payload.
The 36 unlocked historical schema files were removed by the new `contract_lock.py --prune-unlocked` maintenance mode after lock validation.

See [maintained fixture provenance and regeneration](../../../../../contract/fixtures/maintained/README.md).
Focused archive-equivalence, fixture consumers, schema pruning and budget checks run during implementation.
The integration owner runs final cross-language correctness and performance checks after the other Phase 8 efforts merge.
