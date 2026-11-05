<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Recorded client performance

**Historical benchmark, candidate `08d027795`; not a claim about later commits.**

Across these workloads, candidate Python request preparation and schema stages are slower than main, with most of that difference already present in the earlier candidate.
Relational materialize/sample/encode stages are faster in the candidate memory workload; the overall result depends on the workload and host.
All 17 workloads completed; no performance tuning was applied during this campaign.
The measured candidate is artifact revision `08d027795`, compared with main `847a1e9e3` and pre-Phase-6 `4b5df2e2e`.
Later test/tooling changes do not change the measured product paths; the later `labels.py` Arrow 14 compatibility fix (`d140e0d6d`) is outside these workloads.
Python variants share Python 3.12.15, NumPy 2.5.4, pandas 3.0.6, PyArrow 26.0.0 and requests 2.34.2.

Each supported variant uses six fresh processes, each with one cold operation, ten warmups and twenty measured warm iterations.
Numbers below are medians of the six process medians; individual iterations are not treated as independent samples.
Warm time is the sum of measured stages within each iteration, excluding imports, setup, inter-stage assertions and untimed bookkeeping.
It is not full iteration wall latency.

| Warm stage sum (ms) | Main Python | Pre-6 Python | Candidate Python | Candidate Java |
|---|---:|---:|---:|---:|
| Tabular / base | 18.5 | 62.3 | 66.8 | 15.6 |
| Tabular / gradients | 20.1 | unsupported | 68.0 | 14.9 |
| Tabular / 10,000 rows | 130.8 | 158.7 | 160.6 | 56.8 |
| Tabular / 16 features | 39.6 | 129.0 | 135.8 | 20.4 |
| Tabular / real model | 260.3 | 327.1 | 338.8 | 342.2 |
| Relational memory / base | 25.9 | 45.4 | 48.4 | 17.0 |
| Relational memory / gradients | 27.8 | unsupported | 51.7 | 16.6 |
| Relational memory / 10,000 rows | 32.5 | 62.1 | 70.2 | 58.2 |
| Relational memory / 16 features | 31.5 | 82.7 | 87.0 | 26.1 |
| Relational memory / degree 16 | 30.8 | 50.8 | 53.5 | 26.8 |
| Relational memory / real model | 234.2 | 214.8 | 215.5 | 221.1 |
| SQLite / base | 36.4 | 26.3 | 27.6 | 44.0 |
| SQLite / gradients | 37.5 | unsupported | 29.3 | 43.1 |
| SQLite / 10,000 rows | 114.7 | 108.1 | 109.4 | 151.8 |
| SQLite / 16 features | 46.0 | 40.8 | 42.6 | 69.4 |
| SQLite / degree 16 | 58.2 | 45.7 | 46.6 | 67.2 |
| SQLite / real model | 244.5 | 239.2 | 240.0 | 217.3 |

All rows except “real model” use an independent constant-response HTTP fixture; real-model rows use the recorded model endpoint.
Base workloads use 1,000 rows, 10 scored rows, width 4 and relational degree 4; named variants change one dimension to 10,000 rows, width 16 or degree 16.
Gradient cases retain decode, value, count and view checks, but result merge is unavailable for every variant; pre-Phase-6 gradients are unsupported.
Do not compare gradient stage sums with non-gradient sums as identical work.

## Startup, cold work and memory

For the three fixed base workloads, each cell is startup / import / cold stage sum in milliseconds, then sampled peak / time-weighted mean RSS in MiB.
All five quantities are separate medians across processes; Java import time is unavailable.
RSS covers the child client process over its observed lifetime, including imports/setup, but excludes the HTTP server and descendants.

| Base workload / variant | Startup / import / cold (ms); peak / mean RSS (MiB) |
|---|---|
| Tabular / Main Python | 52.2 / 414.6 / 24.4; 122.5 / 102.7 |
| Tabular / Pre-6 Python | 50.9 / 317.6 / 100.6; 127.7 / 117.3 |
| Tabular / Candidate Python | 54.3 / 329.7 / 102.3; 126.6 / 116.5 |
| Tabular / Candidate Java | 82.4 / unavailable / 305.4; 186.2 / 135.5 |
| Relational memory / Main Python | 52.2 / 435.8 / 125.4; 134.4 / 114.6 |
| Relational memory / Pre-6 Python | 51.7 / 301.4 / 85.4; 131.0 / 116.7 |
| Relational memory / Candidate Python | 58.8 / 321.0 / 93.8; 129.0 / 115.8 |
| Relational memory / Candidate Java | 97.8 / unavailable / 322.6; 177.8 / 131.7 |
| SQLite / Main Python | 53.2 / 423.1 / 136.0; 149.8 / 129.2 |
| SQLite / Pre-6 Python | 50.9 / 354.9 / 70.1; 143.0 / 119.0 |
| SQLite / Candidate Python | 54.0 / 323.6 / 64.0; 142.1 / 119.3 |
| SQLite / Candidate Java | 87.4 / unavailable / 428.9; 263.1 / 186.7 |

## Stage observations and comparison limits

In fixed relational memory, main / candidate Python / candidate Java schema stages are 3.1 / 39.7 / 8.0 ms, while materialize/sample/encode stages are 20.5 / 4.6 / 3.9 ms.
In fixed tabular sessions, request preparation is 2.1 / 35.6 / 1.9 ms and score HTTP/decode is 2.4 / 10.4 / 4.0 ms in the same order.
These are observed stage differences, not an attribution to a particular implementation mechanism.
Individual stage medians must not be added to reconstruct the reported stage-sum median.

SQLite does not execute identical sampled context counts across historical and current clients.
In the fixed HTTP lane, recorded main context instances range from 29–32 for base and degree-16, 30–32 for width-16 and gradients, and 31–32 for 10,000-row inputs.
The other variants use 31 context instances, or 32 for 10,000-row inputs; related event counts equal four times that count, or sixteen times for degree-16.
All fixed SQLite prediction requests contain 10 instances and 40 related events, or 160 at degree-16.
These workload differences limit direct speedup interpretation.

Real-model remote cache and startup state are uncontrolled; HTTP request counters are unavailable in that lane.
Python and Java data/view representations differ; these are public client workload comparisons, not isolated Rust or language-runtime measurements.
Native loading is included in import/setup or first operation rather than measured separately.
Startup uses cross-process epoch timestamps, with millisecond resolution in Java.

RSS is sampled every 10 ms, so brief peaks and startup/exit tails can be missed.
The retained evidence includes time-weighted means, sampled peaks, available OS high-water values and process-level ranges.
Its late-window RSS comparison uses the 50–70% and 70–90% portions of process lifetime, not aligned iteration boundaries.
Allocator caching, GC timing and phase mix can affect that proxy; these bounded runs neither establish nor exclude a memory leak.

Evidence is retained outside Git under the retained `benchmark-real` campaign directory: `campaign-summary.json`, `summarize_campaign.py` and each named raw workload JSON.
The summary records raw SHA-256 hashes, artifact revisions, endpoint identities, harness hashes, per-process timings, dispersion, work counters and memory samples; raw results remain unchanged.
