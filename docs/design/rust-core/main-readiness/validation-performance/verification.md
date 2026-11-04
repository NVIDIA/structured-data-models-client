<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Validation checkpoint

**Phase 8 implementation and ordinary acceptance are complete on `rust-core-phase-8`.**
This checkpoint records completed evidence, not a release approval.

## Live behavior

| Scope | Evidence |
|---|---|
| Current Python, both models | All 396 applicable cells pass across 15 connectors |
| Focused live APIs | All 24 Python invocations and seven Java methods pass, including the public S3 reader |
| Current Java matrix | All 396 applicable cells pass; combined Python/Java comparison passes all 792 cells with no unobserved subprocesses |
| Old Python comparison | All 396 cells executed with matched dependencies; 303 strict matches and 93 classified differences |

The matrix and benchmark evidence uses production artifacts built at `08d027795`.
Later CI exposed one product compatibility bug: `query.labels()` used an Arrow helper absent in Arrow 14.
Commit `d140e0d6d` uses the existing capsule-import pattern; its fresh wheel passes 60 focused tests on Arrow 14, with one expected skip.
Prediction benchmarks do not call this labels adapter, so their original artifact identities and measurements remain unchanged.
The baseline is clean `main` commit `847a1e9e3` in separate environments.
Live comparisons match pandas 2.3.3, NumPy 2.5.4 and Arrow 24, and use the same fixtures and recorded model versions.
Earlier observations with different pandas/Arrow versions are diagnostics only.

### Why 93 cells differ

| Cells | Verified difference |
|---:|---|
| 74 | New generated-key provenance metadata; remaining canonical requests match |
| 8 | The same metadata plus the existing exact temporal-reference integer type improvement |
| 1 | The same metadata plus the existing exact decimal-rounding improvement |
| 10 | SQLite context ordering differs; nine historical prediction results also differ |

For all 93, generated-key metadata matches sampled-schema declarations, observed reader rows match with type-sensitive equality, and both requests pass source-membership, unique-node and relationship checks.
Public schemas and lifecycle observations also match.
The strict comparator's original differing verdicts remain unchanged.

The old SQLite sampler warns that it ignores the seed and uses `ORDER BY RANDOM()`.
Repeated public calls confirm that its context order changes, while the Rust client's seeded order repeats.
This supports a reproducibility improvement, not a claim of identical predictions or improved model accuracy.
Independent review checked the classification and the retained raw differences.

## Performance and final local checks

All 17 benchmark workloads completed; two independent reviewers checked the report, raw hashes and all 65 supported warm medians.
See [performance results](performance.md) for all variants, scaling, gradients, real-model timings, startup and memory.
Python preparation/schema stages are slower than old main, with much of the gap already present before Phase 6.
No performance tuning was performed.

The integrated installed-wheel API gate passed 637 cases without skips; 91 tooling integration tests also passed.
The ledger passes against actual main with 240 Python and 534 Java symbols.
Lint, formatting, type checks over 112 source files and repository hygiene pass.
The fresh labels-fix wheel passes 60 focused cases on both Arrow 14 and Arrow 26; each run skips one SQLite-only check's inapplicable DuckDB parameter.

## Evidence and remaining work

Raw observations, artifact identities, dependency inventories, checksums and reproducible classification scripts are retained outside Git in the Phase 8 validation archive.
Key reports are `full-report.json`, `main-s3-comparison.json`, `main-matched-non-s3-comparison.json`, `main-matched-difference-proof.json` and `sqlite-order-repeat-proof.json`.
No credentials or full warehouse captures are committed.

The refreshed AWS login enabled all 52 remaining S3 matrix cells, which pass.
All 26 S3 historical comparisons match strictly.
Dedicated CI identities remain owner-deferred.
Full CI run `41800680419` tests integrated code `24f48e6f7`; later report changes do not alter executable code.
Attempt 2 completed successfully: 81 successful jobs and one intentional targeted-PQL skip.
The only retry was DBR 16.4 after an artifact-download connection reset; its installation and runtime tests subsequently passed.
The focused Java S3 reader test passed without skips using its separate 40-row fixture and verified packaged JNI.
Focused Java tests are source-equivalent runs; the matrix uses the exact recorded JAR.
Independent review verified matrix composition, artifact identities, historical dispositions and scoped plan coverage.
Hosted live CI credentials remain owner-deferred; manual stress execution and release approval are not claimed.
No Phase 8 merge into `rust-core` or `main`, release publication or history rewrite was performed.
