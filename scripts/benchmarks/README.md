<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Client cost measurements

Run this only after the final correctness candidate is integrated.
These tools report measurements; they do not tune performance or enforce a universal regression threshold.
Keep configurations, compiled harness classes, receipts, raw results, and summaries outside the repository.
Install `psutil` in the coordinator environment.
The measured Python environments do not need it.

## Inputs and provenance

Prepare separate environments for pinned main `847a1e9e3`, pre-Phase-6 Rust `4b5df2e2e`, and the final candidate.
Use complete commit hashes in configuration.
Pin common dependencies identically where possible, retain each actual installed dependency inventory, and describe unavoidable differences.
The coordinator verifies clean checkout identity, installed first-party Python source bytes, and native-wheel sidecars against the installed wrapper and extension bytes.
Legacy engine extension hashes are retained separately and verified against the retained built wheel.
For that baseline, set `engine_provenance` to the trusted build receipt containing `commit`, `tree_clean`, `wheel`, `sha256`, and `native_files`; the wheel lives beside the receipt.
Do not present an unverified binary rebuild as source-only Python provenance.
For Java, reuse the live workflow's receipt fields `commit`, `package_hash`, `native_hash`, and `contract`, generated after a clean release build.
The coordinator checks JAR bytes and the packaged JNI library for the current platform, records every dependency JAR hash, and checks which JAR the child actually loaded.
Receipt creation is part of the trusted build; a JSON receipt alone cannot prove how an arbitrary binary was produced.

A private configuration has this shape:

```json
{
  "variants": [
    {
      "label": "main",
      "language": "python",
      "repo": "/isolated/main",
      "revision": "FULL_COMMIT",
      "python": "/isolated/main-env/bin/python"
    },
    {
      "label": "candidate-python",
      "language": "python",
      "repo": "/isolated/candidate",
      "revision": "FULL_COMMIT",
      "python": "/isolated/candidate-env/bin/python",
      "wheel": "/artifacts/client.whl"
    },
    {
      "label": "candidate-java",
      "language": "java",
      "repo": "/isolated/candidate",
      "revision": "FULL_COMMIT",
      "java": "/jdk/bin/java",
      "jar": "/artifacts/client.jar",
      "receipt": "/artifacts/java-provenance.json",
      "dependencies": ["/artifacts/arrow-vector.jar", "/artifacts/other-required.jar"],
      "harness_dir": "/artifacts/benchmark-classes"
    }
  ]
}
```

Add the pre-Phase-6 variant with its own environment, revision, and native wheel.
The wheel sidecar must be named `<wheel>.json`, as produced by `native_wheel.py`.
An optional `dependency_sha256` must match the actual environment inventory digest; it is checked, not merely echoed.
List all Java runtime dependency JARs explicitly, including SQLite JDBC for that workload.
The coordinator compiles `ClientCost.java` against the verified Java artifact and dependencies into `harness_dir` before measuring, using `javac` beside the configured `java` executable.
This binds the reported harness class hash to the current source rather than trusting stale compiled classes.
Source-file launch would include Java compilation in the cold measurement and is intentionally not used.

## Workloads and repetition

```bash
python scripts/benchmarks/client_cost.py \
  --config /private/variants.json --out /private/tabular.json \
  --workload tabular-session --rows 1000 --scores 10 --width 4 \
  --processes 6 --warmups 10 --loops 20
```

Run `relational-memory` and `relational-sqlite` with the same configuration and dimensions.
Vary one of rows, width, or degree at a time using bounded values supported by all compared clients.
Inputs are generated from identical formulas in Python and Java; no large fixture is retained.
For Tabular sessions, `rows` is the context size and `scores` adds disjoint scoring IDs starting at `rows`; no unused event graph is allocated.
For relational workloads, `rows` is the users-table size, the last `scores` users are queried, and the events table has `rows × degree` rows.
Each child reports these actual input row counts separately from the request-row counts observed by the HTTP fixture.
The relational query predicts the entity's numerical target with one-hop related events, explicit keys/relationships and numerical target/feature types, seed 7, and at most 32 context examples.
The SQLite route uses ADBC in Python and JDBC in Java, and preserves each driver's ordinary representations.
It includes real source schema discovery and SQL pushdown; it is not a reader-normalized memory substitute.
Tabular sessions measure request preparation, fit, score, close, result conversion, and merge.
All workloads measure result merge and check independent expected row counts and fixed-response values.

Repeat each workload with `--gradients` for clients exporting the public gradient view.
The relational fixture assigns gradients to the known `users.x*` numerical features, using actual related-table positions filtered by each prediction’s `instance_id`.
It omits gradients for other tables rather than inventing values or interpreting every feature as numerical.
Both hosts require nonempty gradient views and record their cell counts for comparison.
The fixed HTTP fixture independently constructs regression gradients of 0.25 with correct row/output/column dimensions, and the clients check gradient values and merge identities.
A historical client lacking the gradient API reports unsupported; it is not counted as a successful zero-cost measurement.
This measures gradient response handling, not model differentiation or model quality.

Each subprocess performs a distinct cold first iteration, explicit warmups, then retained warm iterations.
Variants run in forward/reverse order on alternating process repetitions.
Neither runtime is forced to collect garbage.
Use a quiet machine, inspect warm sample stability and JVM warmup, and repeat the measurement with more warmups when required.
Keep raw process repetitions; summarize warm per-stage medians within each process, then compare those independent process medians.
Do not treat loops in one process as independent cold samples.

## What each number means

`startup_to_harness_entry_seconds` covers process creation through the child harness entry using the same machine's wall clock.
Python records imports separately; Java class loading, native loading, and JIT work fall into setup or the first public operation.
Native load is not independently instrumented in either host.
`cold-first` stage times exclude import/setup and must remain distinct from total startup cost.
`process_wall_seconds` includes startup, setup, all iterations, validation, and shutdown, so it is not a single-operation latency.
Stage timers use monotonic clocks and do not overlap.
`materialize_sample_encode` combines all work inside the public materialize call.
Python result views produce pandas frames, while Java result views read Arrow batches; both validate the same answers but have different host conversion costs.
The HTTP fixture runs in the coordinator process and uses the same response policy for every client.
Accepted fixture sockets disable Nagle buffering so separate header/body writes do not add a delayed-ACK pause.
HTTP call and payload-byte counters include cold, warmup, and warm iterations.
Each HTTP request also records context/predict instance-row counts and related-table row counts, without retaining payloads.
Compare these work counts before attributing a timing difference to implementation cost; sampling differences must remain visible.
The pinned old client's SQLite sampler ignores the seed, so its sampled context count can differ after scoring rows are excluded.
Report those actual context/related-row counts alongside latency; do not describe that comparison as identical sampled work or attribute the entire difference to engine speed.
Native call counts and SQLite query counts are unavailable without additional instrumentation and are explicitly null.
Memory-route SQL counts are zero.
No correctness trace listener or attachment-copy instrumentation is enabled.

RSS samples cover the benchmark process only, excluding the fixture server and descendants.
Observed descendant PIDs are reported so unexpected subprocess work cannot be silently treated as included memory.
The sampled peak can miss short spikes; it is distinct from Linux Python's process RSS high-water mark.
The mean uses trapezoidal time weighting across actual sample intervals, with unsampled startup/exit tails excluded explicitly.
Raw RSS samples support inspection for continuing retention over repeated close/disposal cycles.
Imports/setup are included in process memory, not misreported as isolated native allocations.

## Real model lane

Use `--real-config /private/endpoint.json` containing `url` and `model_identity` to replace the fixed fixture with an already-provisioned endpoint.
No model or GPU is deployed by this tool.
Use an exact model/version identity verified by the acceptance campaign and retain that evidence alongside the results.
The recorded endpoint identity comes from this configuration; the benchmark does not independently attest a remote model.
The URL is passed through the child environment and is not written to results.
Client HTTP counters are unavailable in this mode.
Keep remote cache/startup state and model CPU/GPU memory separate from client measurements.
Constant-value assertions apply only to the fixed fixture; real-model runs retain row-count and merge consistency checks.
