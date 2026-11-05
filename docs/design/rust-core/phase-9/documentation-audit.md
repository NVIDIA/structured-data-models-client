<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Migration documentation cleanup

**Audit and proposal only; no files deleted.**
Inventory checked on `rust-core-phase-9`, based on `0b3afe8fa`.
Reviewed file inventory, references outside the migration folder, diagram source/text, current developer guidance, date specification and verification reports.
This is a documentation cleanup, not permission to remove runtime contracts or tests.

## Recommendation

Remove completed migration plans after extracting the few still-useful contracts, evidence and follow-ups into maintained documentation.
Use Git history for the old plans rather than creating another archive directory in the repository.
The existing folder is about 3 MB: this primarily improves clarity, not disk usage.

| Files audited | Proposed disposition |
|---|---|
| `phase-5/`: 21 Markdown files, 3 API-audit CSVs | Remove after checking that enduring API decisions are represented in current API docs, the compatibility manifest and tests. Preserve unresolved limitations, not old implementation trackers. |
| `phase-6/`: 22 Markdown files | Remove completed restoration plans, commit inventories and completion logs after the same contract check. Move any still-open follow-up into one maintained list. |
| `main-readiness/`: 11 Markdown files | Remove superseded plans and merge inventories. Consolidate current CI policy, exact-commit validation evidence, benchmark results and outstanding limitations before removing originals. |
| `docs/design/date-rules.md` | Keep the current parsing specification, shorten historical Python investigation, and move the maintained rules to a clearly named reference page. This is not merely a migration diary. |
| `images/`: 4 SVGs, 4 PNGs, generator | Keep/update useful architecture and bindings diagrams; remove obsolete delivery and speculative PQL diagrams. Details below. |
| `phase-9/design.md` | Keep while this work is active; promote lasting decisions and remove the plan when complete. |

## Preserve the useful pictures, correct their claims

| Diagram | Finding and action |
|---|---|
| `architecture` | Useful host/core/I/O overview. Update external PQL, example-provider and future-capability labels against current code before moving to developer docs. |
| `bindings` | Useful PyO3/JNI/host boundary. Remove claims that external C++ PQL is linked and Java only follows Python; qualify zero-copy claims by actual path. |
| `pql` | Explicit future external C++ integration proposal, not the current `sdm-pql` parser. Delete from current architecture documentation; history retains the proposal. |
| `delivery` | Phase-by-phase migration schedule and obsolete CI/release assumptions. Delete. |

Keep editable diagram source and one rendered format supported by the documentation renderer; retain PNG alternatives only where actually needed.
Update `make_diagrams.py` and regenerate outputs, never hand-edit generated SVGs/PNGs.
Link the retained diagrams from `rust/README.md` and the developer change guide so maintainers can find them.

## Concrete dependencies and stale statements

- Rust time parsing, schema inference, PQL dates, contract comments and date regression vectors cite `date-rules.md`.
  Its current format rules and PQL literal rules must survive; source references must follow the move.
- Root `README.md` still calls Phase 5 the current completion plan; replace that link with maintained developer guidance.
- `tests/parity/README.md` points to the Phase 5 verification tracker, and the manifest has a historical visualization-plan citation.
  Replace these with maintained coverage/behavior references without rewriting historical test outcomes.
- Older verification pages say CI credentials are deferred, whereas the newer hosted acceptance report records successful hosted runs.
  Consolidate chronologically, preserving tested SHAs and scope rather than suggesting every result applies to current HEAD.
- Generated contract snapshots also contain historical date-rule citations.
  Do not hand-edit them or regenerate historical schemas merely to clean prose; update current source documentation and use the normal generator only where needed.

## Small maintained destination

| Existing home | Keep there |
|---|---|
| `rust/README.md` and crate READMEs | Engine boundaries, operation flow, corrected architecture/bindings pictures. |
| `docs/development/change-guide.md` | How to add/change APIs, ownership rules, matrix/test enforcement; link existing machinery rather than duplicate it. |
| `tests/parity/README.md` | Test layers, credential requirements, matrix interpretation and CI triggers. |
| New `docs/reference/date-rules.md` | Current date parsing contract with links to implementation and regression tests. |
| New `docs/development/validation.md` | Compact dated evidence and benchmark tables, tested commits/run links, known limits and remaining follow-ups. |

Preserve distinctions between materialization coverage and model inference, emulated versus real S3, excluded Lakebase, deferred Windows ARM64 and unexecuted acceptance gates.
Retain benchmark methodology and measured regressions; do not discard inconvenient results or treat old evidence as Phase 9 certification.
CI parallelization remains a post-main follow-up.

## Cleanup gate

First extract and verify current contracts against code/tests; then delete historical documents and fix incoming links.
Check Markdown links, diagram rendering, repository hygiene and contract references; no full model/warehouse rerun is needed for documentation-only changes.
Do not remove test cases, manifests, fixtures, generated contracts or benchmark machinery as a side effect of this cleanup.
