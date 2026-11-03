<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 4. RelBench dataset loading

**Status: assigned for contract-first implementation. Support both Python and Java; use their existing file/Arrow readers.**
Load a named, pinned dataset into the maintained `Schema`, with its keys, relationships and time metadata.
This restores a convenience loader, not a second sampling path.
Independent boundary review is incorporated below; this document is the implementation contract.

```text
Dataset name → common pinned manifest
                        ↓
Python / Java download, verify hash, cache, read Parquet
                        ↓ Arrow + declared metadata
           existing shared Rust schema admission
                        ↓
                 Python / Java Schema
```

| Layer | Responsibility |
|---|---|
| Common packaged manifest | Dataset names, immutable revisions, file hashes, table/key/link/time metadata and declarative exclusions/type overrides |
| Python / Java loader | Download/cache I/O and Parquet reading using existing readers; Java already has a Parquet reader |
| Shared Rust schema admission | Apply declared metadata and validate tables/relationships; any nontrivial dataset transformation must have shared semantics |

## Proposed user API

`Schema.from_relbench(name, cache_dir=...)` / `Schema.fromRelbench(name, options)`.
Both return ordinary schemas usable through the existing Relational workflow.
Use one canonical manifest packaged into wheel and JAR; do not maintain separate dataset registries.
Recover the old pinned registry and verify its entries before reuse; do not silently switch users to a changing upstream revision.
Do not require a Python-only RelBench dependency for Java compatibility.

Define cache hits, offline cache use, integrity failures and incomplete-download recovery consistently.
Stream downloads to temporary files, verify hashes, then atomically install them in the cache.
Reject unsafe relative paths and preserve declared dataset-specific exclusions and type overrides.
Respect existing Arrow ownership conventions; Java schema data must remain usable after temporary readers close.
No Rust HTTP/cache subsystem, model call or warehouse sampling changes are needed.

## Acceptance

Use a small local HTTP/Parquet fixture to test both loaders, cache reuse, hash mismatch and interrupted downloads.
Compare rows, declared schema, keys, relationships and time columns across languages.
Validate each manifest entry structurally and perform a bounded pinned real-dataset smoke check; CI must not download the whole dataset catalog.
Run one downstream prediction smoke when implemented to verify the resulting schema integrates with the existing path.

## Implementation and review gate

First finalize the public signatures, result shapes, ownership and error behavior here; obtain coordinator contract review before implementation.
Add focused public-API and cross-language tests first where practical, then implement the reviewed behavior.
Use existing parity/snapshot tooling; do not hand-edit generated snapshots or introduce migration-only machinery.
Finish with independent code review, address findings, and run affected tests before merging into `rust-core-phase-6`.

Keep exclusions and type overrides declarative in the common manifest; do not encode independent dataset policy in each language.
