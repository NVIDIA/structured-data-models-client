<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Validation checkpoint

**Phase 8 is not yet accepted: Java matrix completion, corrected platform CI, S3 and performance measurements remain pending.**
This checkpoint records completed evidence, not a release approval.

## Live behavior

| Scope | Evidence |
|---|---|
| Current Python, both models | All 370 applicable non-S3 cells pass across 13 connectors |
| Focused live APIs | All 24 Python invocations and six available Java methods pass; Java S3 is unavailable |
| Current Java matrix | Rerun in progress after a reviewed authentication-observation repair |
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

## Evidence and remaining work

Raw observations, artifact identities, dependency inventories, checksums and reproducible classification scripts are retained outside Git in the Phase 8 validation archive.
Key reports are `python-non-s3-final-report.json`, `main-matched-non-s3-comparison.json`, `main-matched-difference-proof.json` and `sqlite-order-repeat-proof.json`.
No credentials or full warehouse captures are committed.

S3 credentials remain expired; dedicated CI identities are owner-deferred.
Benchmarks have only passed harness smoke checks so far; smoke timings are not performance results.
Final measurements must report imports/cold/warm conditions, actual sampled work and process-memory scope, without performance tuning.
