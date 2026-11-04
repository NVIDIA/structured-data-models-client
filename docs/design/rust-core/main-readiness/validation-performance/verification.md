<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Validation checkpoint

**Phase 8 is not yet accepted: corrected platform CI, S3 and performance measurements remain pending.**
This checkpoint records completed evidence, not a release approval.

## Live behavior

| Scope | Evidence |
|---|---|
| Current Python, both models | All 370 applicable non-S3 cells pass across 13 connectors |
| Focused live APIs | All 24 Python invocations and six available Java methods pass; Java S3 is unavailable |
| Current Java matrix | All 370 applicable non-S3 cells pass; combined Python/Java comparison passes all 740 cells with no unobserved subprocesses |
| Old Python comparison | All 370 non-S3 cells executed with matched dependencies; 277 strict matches and 93 classified differences |

The candidate uses production artifacts built at `08d027795`; subsequent changes repair validation tooling and tests, with unchanged product source.
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

## Initial performance results

These are fixed-response client measurements, not model latency.
Each number is the median of six independent process medians of summed timed warm stages, in milliseconds.
Each process has one cold iteration, ten warmups and twenty measured iterations.

| Workload | Old main Python | Pre-Phase-6 Python | Current Python | Current Java |
|---|---:|---:|---:|---:|
| Tabular fit → predict → close | 18.47 | 62.27 | 66.75 | 15.59 |
| Relational in-memory prediction | 25.91 | 45.41 | 48.39 | 17.02 |
| Relational SQLite prediction | 36.45 | 26.35 | 27.64 | 43.99 |
| Tabular with gradients | 20.05 | Unsupported | 68.00 | 14.94 |
| Tabular, 10,000 context rows | 130.85 | 158.70 | 160.57 | 56.82 |

Unless the row specifies otherwise, inputs use 1,000 rows, ten scoring rows and four features; relational degree is four.
Independent review confirmed matching request work counts and common Python dependencies.
The largest Python differences occur before HTTP, in request preparation and schema construction; most of the gap already exists before Phase 6.
These are substantial regressions, reported without performance tuning as requested.
Java uses Arrow result views and Python uses pandas, so host conversion costs differ.
SQLite's old sampler produced 29–32 context rows versus 31 in the Rust clients; its timings therefore do not represent identical sampled work.
Gradient timing excludes explicit result merge in both hosts because the old API does not support the same repeated-result merge workload.

Import/startup and cold timings remain separate from this table.
Memory samples cover the whole process lifetime, including imports, rather than warm-only or isolated Rust memory.
The retained raw results are `benchmark-real/fixed-tabular-session.json` and `benchmark-real/fixed-relational-memory.json` in the external Phase 8 archive.
Further width/degree scaling, relational gradients and real-model measurements remain pending; these results do not establish overall performance.

## Evidence and remaining work

Raw observations, artifact identities, dependency inventories, checksums and reproducible classification scripts are retained outside Git in the Phase 8 validation archive.
Key reports are `non-s3-validated-report.json`, `main-matched-non-s3-comparison.json`, `main-matched-difference-proof.json` and `sqlite-order-repeat-proof.json`.
No credentials or full warehouse captures are committed.

S3 credentials remain expired; dedicated CI identities are owner-deferred.
Benchmark harness smoke checks passed; the remaining measured campaign is running, and smoke timings are not performance results.
Final measurements must report imports/cold/warm conditions, actual sampled work and process-memory scope, without performance tuning.
