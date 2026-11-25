<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Validation evidence and open limits

This page records tested revisions and scope; it does not certify current HEAD or authorize publication.
For commands and current CI policy, see [CONTRIBUTING](../../CONTRIBUTING.md) and the [compatibility suite](../../tests/compatibility/README.md).
The compatibility manifest and execution receipts remain authoritative for individual capabilities.

## Recorded evidence

| Revision / campaign | Result and scope |
|---|---|
| Rust source `59addf5cc` | Host AddressSanitizer/leak check: 128 tests across 18 executables, no ignored tests or diagnostics; nightly-2026-07-15 with rebuilt standard library. Not Python/JVM sanitizer coverage. |
| Candidate `08d027795`, baseline main `847a1e9e3` | 792 local live cells, 396 per language across 15 connectors; same fixture/model versions. Old Python comparison: 303 strict matches and 93 classified differences. |
| `d140e0d6d` labels compatibility fix | Fresh wheel passed 60 focused tests on Arrow 14 and Arrow 26, each with one expected inapplicable parameter skip. Prediction benchmarks do not exercise this adapter. |
| `24f48e6f7`, [CI 41800680419](https://github.com/NVIDIA/structured-data-models-client/actions/runs/41800680419) | Attempt 2: 81 successful jobs and one intentional targeted-PQL skip. DBR16.4 retried after artifact-download connection reset. |
| `3cb611a051dabff024f7847c5e00473e18b313bf`, [CI 41929869913](https://github.com/NVIDIA/structured-data-models-client/actions/runs/41929869913) | All 81 jobs passed; 32.9 minutes including scheduling. |
| Same `3cb611a…`, [hosted 41934656118](https://github.com/NVIDIA/structured-data-models-client/actions/runs/41934656118), 2026-11-05 | All 17 jobs passed; 742/742 materialization cells across 14 connectors, both languages/models; 38.0 minutes including scheduling. Separate pretrained CPU smoke passed four Python cases and three Java methods in 55.9 seconds. |

The hosted run verified all 14 cleanup receipts and matching production artifact identities; its downloaded aggregate was independently replayed successfully.
Java connector journeys loaded the packaged JAR; Java CPU smoke compiled the same checkout and used the verified packaged JNI library.
Its `hosted-parity-3cb611a051dabff024f7847c5e00473e18b313bf` artifact contains aggregate observations; shards retain execution/build receipts.
These hosted results supersede older statements that dedicated CI identities were unavailable.

### Historical differences are not erased

Of the 93 differing old-Python cells, 74 differ in generated-key provenance, eight add exact temporal-reference integer types, one adds exact decimal rounding, and ten differ in SQLite context ordering.
Nine of the SQLite prediction results also differ; old sampling used unseeded `ORDER BY RANDOM()`, while repeated seeded Rust calls were reproducible.
This establishes reproducibility, not identical predictions or better model accuracy.
For all 93, type-sensitive reader rows, public schemas and lifecycle observations matched, with request source-membership and relationship checks.
Strict comparator difference verdicts remain unchanged.
The live comparison used matched pandas 2.3.3, NumPy 2.5.4 and Arrow 24; observations with other dependencies were diagnostic only.

## Performance and acceptance limits

[Performance results](performance.md) retain all 17 workloads, warm/cold/startup measurements, RSS, source identities and comparison limitations.
Candidate Python preparation/schema stages were slower than old main; relational memory sampling/encoding was faster.
No benchmark-driven tuning was performed.
Historical accepted costs include approximately 2× Databricks latency, 21–32% Snowflake overhead, native library size around 16 MB versus 9.6 MB, and specific SQLite cases; these are scoped decisions, not blanket acceptance of future regressions.

| Limit / follow-up | Required interpretation |
|---|---|
| Hosted matrix versus inference | Broad cells test read/materialization; actual inference is separate bounded CPU smoke. Do not label all cells live-model predictions. |
| Lakebase / AWS | Hosted excludes 50 Lakebase cells and uses S3 emulation. Earlier local real-S3 validation passed 52 cells and 26 strict historical comparisons; emulation cannot certify real AWS. |
| Release and stress | Exact-release-commit ordinary/live and stress gates still apply. Manual stress execution and release approval were not claimed by these reports. |
| Platform scope | Linux x86-64/ARM64, macOS x86-64/ARM64 and Windows x86-64 are the five distribution targets. Native Windows ARM64 Python/Java remain deferred for upstream Arrow availability. |
| Historical acceptance | No claim that the final candidate replayed every old refusal-wording census, original randomized case or runtime sweep. Carry-forward requires artifact/scope review, and affected cases must be rerun. |
| Optional text summarizer | Local HTTP contract verified; real text-model prose quality was not validated. |
| CI duration | The 10-minute aspiration remains unmet. Warm Rust validation fell from 29.5 to 15.4 minutes, with an 11.7-minute test step. Spark took 17.0 minutes; Databricks SQL 7.1, metric views 6.1, Snowflake 2.6–3.4, excluding setup/queueing. |

After merging into main, review larger runners, safe sharding, connector concurrency and artifact reuse for unchanged build inputs.
The recorded hosted configuration used standard GitHub-hosted Linux runners and two concurrent connector shards.
Subsequent changes need their own affected-behavior evidence; these older passes are not inherited automatically.

## Evidence retention

Historical plans remain in Git at `0b3afe8fa:docs/design/rust-core/`; no extra in-repository archive is needed.
Private receipts and raw model/warehouse observations remain outside Git under `~/.local/share/sdm-orchestration/`; they are not portable public CI artifacts and must not be silently deleted during cleanup.
The benchmark summary records raw SHA-256 hashes, harness/artifact identities, per-process timings, work counts and memory samples.
Retain unmodified receipts; never commit credentials or full warehouse captures.
