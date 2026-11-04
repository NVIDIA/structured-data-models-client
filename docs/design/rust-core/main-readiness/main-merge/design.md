<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Merge main without restoring the old engine

**Do this before the other readiness work: merge main's contracts and improvements into the Rust architecture, then prove which paths execute.**
This is a plan; the merge has not been applied to `rust-core`.

## What we audited

`rust-core` is `d184fb269`; `origin/main` is `847a1e9e3`; their merge base is `12955d397`.
Only PR #34 (gradients) and PR #62 (EXPLAIN) are missing from our ancestry.
They touch **43 files**: the simulated merge changes 39, with **20 conflicts and 19 clean merges**; four files already absorb main's changes.
The [inventory and operation map](files.md) covers all 43.

An isolated merge also proved Python would import main's new `explanations/` package instead of our `explanations.py`.
Git reports no conflict for that collision.
Neither “accept ours everywhere” nor accepting clean merges automatically is safe.

## Missing behavior found during this audit

The first four findings were reproduced through the installed Phase 6 public Python API.

| Finding | Required resolution |
|---|---|
| `PredictionResult(..., explanations=...)` raises `TypeError`; main supports user-constructed results | Restore construction and merging; Python owns containers, Rust validates row/step identity and merge rules. |
| `EXPLAIN` with `outputs=[]` sends an explanation request | Reject the empty list in Rust before EXPLAIN augmentation; assert zero model calls. |
| Unsupported in-process output attempts model loading first | Check runtime capabilities before loading in Python's model adapter; keep result interpretation in Rust. |
| EXPLAIN candidate ranking materializes successfully, then fails at prediction | Reject EXPLAIN/ranking in Rust before source gathering. Outputs supplied later to `predict()` still validate there. |
| Generated related-table keys may appear as user keys | Verify keyless tables and genuine user `__node_id...` columns first. Track generated-key provenance in Rust instead of copying main's prefix heuristic. |

Retain strict errors, typed identities, nested merges and the singular raw view.
Proposed construction policy: restore Python’s existing convenience API; retain Java’s current constructor visibility and record that difference in the ledger.
Java-produced results still use the same Rust merge rules.
Do not restore main's already-expired “removed in 2.0” deprecation deadline.
Forecast missing-step warnings and session EXPLAIN → plain reset already exist; verify their structured-gradient variants rather than reimplementing them.

## The boundary we must preserve

```text
Python / Java public calls
    → Rust: query/output policy, preparation, decoding, identity and merging
    → host: database/HTTP I/O or optional Python model execution
    → Rust: validated results
    → Python / Java: containers, pandas/Arrow views and rendering
```

Main's Python `GradientDecoder`, adapter `answered()` and EXPLAIN regex helpers must not become alternative engine paths.
Keep `predictive_query` and the retired relational-engine runtime absent.
Tensor conversion remains Python: main's converter is already ported as `in_process/gradients.py`.
Preserve supported native Python object handling through existing Rust-directed operations; a compatibility fallback is not automatically a legacy path.

## Three implementation steps

1. **Resolve by behavior on an isolated merge branch.**
   Refresh and pin `origin/main`, update the inventory if it moved, then merge without committing.
   Retain Rust-owned behavior, port the findings above and useful main documentation/assertions, and keep retired packages deleted.
   Map every incoming public contract to a test or reviewed difference.
   Regenerate API/native-contract snapshots; never hand-merge them.
2. **Prove routing as well as outputs.**
   Extend `tests/parity/routes.py` for the newly introduced Python decoder and EXPLAIN/adapter/decomposition helpers.
   Add installed-wheel checks for ambiguous module/package names, export origins and retired modules/dependencies.
   Require the expected Rust operations for each journey, not merely a loaded native library or nonzero call count.
   Prove the gates fail on forbidden helpers and missing Rust operations.
3. **Validate and integrate only the reviewed merge.**
   Adapt main's useful assertions to public requests/current test seams, without restoring legacy hooks.
   Run affected suites, schema/API/ledger gates, installed wheel/JNI checks and real gradient-capable models in both languages.
   Review both-parent diffs; merge into `rust-core` only after checks pass and `origin/main` is an ancestor.

## Required route tests

Apply these to Python and Java where supported; local model execution is Python-only.

| Journey | What must be demonstrated |
|---|---|
| Direct and PQL prediction | Correct Rust preparation/decoding; empty-output refusal; no Python decoder |
| Frame/warehouse and Tabular sessions | Rust fit/score/close operations; explained then plain calls; cleanup on failure |
| Forecast and ranking | Structured gradients and step identities, missing-step diagnostics, ranking refusal before source work |
| User-built and nested results | Matching rows, duplicate IDs/steps, orphan-key refusal; shared Rust merge rules |
| Malformed and native-object responses | Predictable errors, preserved allowed ownership, no swallowed gradient error or legacy fallback |
| Local model runtime | Valid tensor execution; invalid output rejected without loading; returned data interpreted by Rust |

Use the existing fixtures and guards; no new migration framework or full capture archive.
A passing output comparison alone is insufficient if it could have run through main's Python engine.
Size cleanup, broad CI redesign and benchmarking remain subsequent work.
