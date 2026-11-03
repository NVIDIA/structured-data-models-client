<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 4. RelBench dataset loading

**Status: implemented and tested; coordinator review before integration.**
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

## Approved public contract

| Python | Java |
|---|---|
| `Schema.from_relbench(name, *, cache_dir=None, offline=False)` | `Schema.fromRelbench(name)` and `Schema.fromRelbench(name, RelBenchOptions)` |
| `cache_dir` accepts a path/string | `RelBenchOptions.builder().cacheDirectory(path).offline(true).build()` |

Both return ordinary, independently owned schemas usable through the existing Relational workflow.
Names are case-insensitive; `amazon` and `rel-amazon` select the same pinned dataset.
The default cache is `~/.cache/sdm/relbench`; explicit paths allow shared caches.
Cache entries use repository/revision/dataset/table paths and are hash-checked on every load.
Offline mode never opens a network connection and rejects missing or corrupt entries.
Online mode replaces missing/corrupt entries only after streaming to a unique temporary file, checking SHA-256 and atomically replacing the destination.
A failed download never replaces a valid entry and removes its temporary file.
Unknown names/options use `INVALID_REQUEST`; download, cache and integrity failures use `PROVIDER_ERROR`, with dataset/table context.
Existing schema-validation errors remain unchanged.
Loading is quiet by default.
A null key/time declaration means explicitly absent; inferred relationships are cleared before metadata edits and replaced only with declared links.

## Boundaries and evidence

| Evidence | Implementation decision |
|---|---|
| Old `rfm/relbench.py` loads pinned Parquet, then applies keys/time/links and exclusions/type overrides | Restore declarations, without recreating the old graph or sampling engine |
| Old registry identifies immutable Hugging Face commits and SHA-256 hashes | Restore generated registry byte-for-byte from `origin/main`; retain its generator with updated output location |
| Old exclusions cover HM/Amazon; explicit type overrides cover SALT | Keep these in a small hand-authored JSON overlay alongside the unchanged generated registry |
| Python `Schema.from_data` accepts explicit key/time/link metadata | Read Parquet using existing PyArrow dependency, then use this entry point and public metadata setters |
| Java `Readers.read` reads Parquet; `Schema.fromData` takes owned snapshots | Use these APIs and close temporary Arrow tables/allocator before returning; apply metadata through existing schema edits |

The canonical registry and overlay live in Python package resources; Maven packages those same files in the JAR.
Do not require the archived engine, `relbench`, or `pooch` dependencies.
The loader performs only declarative column exclusion and metadata application; Rust retains schema validation, inference and all sampling.
Reject unsafe manifest path components before filesystem/network access.
No Rust HTTP/cache subsystem, Python callbacks, model calls or warehouse sampling changes are needed.

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

## Validation result

- Python loader + parity ledger/contracts: 58 passed, two existing environment-dependent skips.
- Java loader: four tests passed, including shared fixture rows, metadata, ownership and HTTP recovery; JAR contains the exact shared catalogs.
- Pinned `rel-f1`: 1,224,756 bytes downloaded, nine tables, 97,606 rows and 13 links; Python and Java counts match and schema checks pass.
- Raw full-dataset prediction reaches the existing v27 endpoint but is refused with `UNSUPPORTED_DTYPE`, consistent with its documented unsupported-text behavior.
- Explicitly declaring F1 name/reference text fields categorical through the public schema API produces a successful two-row regression prediction, with 32 context samples.

The loader preserves benchmark metadata rather than silently changing types to satisfy a particular model.
No core or warehouse code changed.
Independent review identified Java dictionary-owner lifetime and interruption handling; both are addressed and retested.
