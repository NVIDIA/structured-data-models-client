<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Final correctness and performance checks

**Proposal: validate the maintained matrix on the final candidate, then benchmark changed paths with a small repeatable suite.**
Phase 6's focused tests do not establish whole-matrix correctness or performance.
This is a plan, not a new test or benchmark result.

## Existing evidence

| Evidence | Reuse and limitation |
|---|---|
| Phase 5 CI/live receipts | Valid for their recorded artifacts and scope; review changed code/dependencies before reuse |
| Phase 6 gradients | Real classification/regression in both models/languages passed; focused session, forecast and error tests passed; not every connector or resource profile |
| `tests/parity` runners/comparator | Reuse public journeys, typed input comparison and native-route checks; the live API registry still needs execution wiring |
| Earlier session benchmark | Fit 23.9 → 27.4 ms, score 6.4 → 7.3 ms: about +15%, accepted for that RC; no equivalent Phase 6 campaign |

That benchmark used 10,000 context rows, 100 scoring rows and a constant-response transport.
Its peak RSS included imports/setup; neither network/model latency nor isolated native memory was measured.
Sources: [Phase 5 verification](../../phase-5/verification.md), [Phase 6 evidence](../../phase-6/main-parity/final-three-implementation.md), [runner guide](../../../../../tests/parity/README.md).

## Correctness scope

```text
Same fixture + question + options + pinned model
       |                 |                 |
 main Python       current Python     current Java
       +-------- inputs, results, lifecycle --------+
                         |
          expected contract + reviewed differences
```

Extend the existing manifest with missing Phase 6 compositions, then run every applicable declared scenario on the final candidate.
Use raw read → predict without test-specific cleanup; unsupported and unavailable combinations remain explicit.

| Changed area | Priority checks |
|---|---|
| End-time, expressions, task/type edits, time series | Matching labels/rows across memory and applicable SQL dialects; null/time boundaries; no extra scans |
| Rebinding, batching, retries, sessions | Equivalent sources, expiry/recovery, partial failures and cleanup; bounded model/query counts |
| DuckDB 2, PostgreSQL ordering | Supported real drivers; JSON/XML/numeric/null/time inputs; borrowed connection ownership |
| Gradients, reports, rendering | Off/on, multiple rows/steps/merges, malformed output; no unnecessary copies or retained owners |
| Loaders, persistence, conversions | Both hosts; decimal keys, nullable free text, timezone timestamps and supported list/stype combinations |

Preserve unique, valid type coverage while shrinking fixtures per the [size plan](../repository-size/design.md); add connector-specific cases.
Use shared engine tests for common rules and both host tests for conversion/ownership.
Compare overlapping behavior with pinned current `main` Python in an isolated environment.
The historical comparator currently hardcodes baseline `43e55f4f5` in `tests/parity/contract.py`; add a reviewed explicit baseline/provenance parameter before reusing it for current `main`, preserving archived receipts.
Java has no old counterpart: compare it with current Python and independent expected outcomes.
Keep the baseline outside the maintained checkout; do not preserve old bugs or restore the retired runtime to CI.

## Benchmark what matters

```text
startup → read/convert → schema/PQL → sample → encode → model I/O
                                                          |
                                      close ← render ← decode/merge
```

| Metric | Measurement |
|---|---|
| Cold latency | Fresh Python/JVM process, native load and first request/result; model startup separately |
| Warm latency | Repeated requests and fit/predict/close, with explicit warmups and cache/JVM JIT state |
| Per-step latency | The stages above plus native/provider calls; label overlapping versus exclusive intervals |
| Peak/average memory | Fresh-process RSS high-water and time-weighted sampled RSS; record units, sample interval and child-process scope |
| Retention/scaling | Repeated close/disposal; distinguish bounded allocator caching from continuing growth; vary rows, width and graph degree |
| Work counters | Queries, native/model calls, rows and bytes moved |

Run a fixed-response lane for client cost and a real-model lane for end-to-end latency.
Keep local model CPU/GPU memory separate from endpoint-client memory; record remote cache/startup state as uncontrolled when necessary.
Use deterministic larger inputs within supported limits, generated at runtime rather than committed.

## Keep the tooling small

Reuse case builders and installed wheels/JARs.
Use [pyperf](https://pyperf.readthedocs.io/en/latest/cli.html) for repetitions/stability/comparisons and [psutil](https://psutil.io/) for process RSS, including native Rust/Arrow memory.
Cold Java runs can use command mode; warm measurements must stay inside a warmed JVM.
[Python allocation tracing](https://docs.python.org/3/library/tracemalloc.html) alone is insufficient for total process memory.
Java's existing `internal/Trace.java` copies Arrow attachments: keep correctness captures off during performance runs.
Add lightweight stage timers/counters, not a second capture framework; use a profiler for deeper attribution.

Start on a quiet reference machine, alternate baseline/candidate runs and retain sample distributions before choosing thresholds.
Compare against both current `main` and pre-Phase-6 Rust `4b5df2e2e`, pinning dependency environments and separating dependency effects.
New-only features need scaling/absolute budgets; the old accepted 15% cost does not authorize further regressions.

## Completion

1. Map changed paths to cases, reuse decisions and executable live requirements using the [CI plan](../ci-coverage/design.md).
2. Run correctness and bounded benchmarks; fix demonstrated issues and repeat affected checks.
3. Validate final platform artifacts and the full applicable live matrix; retain compact summaries and checksummed evidence outside Git.

Before `main`, required correctness cells must pass or have explicit reviewed dispositions; unavailable resources stay incomplete.
Performance requires stable per-scenario time/memory budgets or reviewed exceptions, not an arbitrary universal percentage.
