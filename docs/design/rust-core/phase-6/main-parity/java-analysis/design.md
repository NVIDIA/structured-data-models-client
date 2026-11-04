<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Include Java in static security analysis

**Gap:** shipped Java code is not in the CodeQL language matrix.
This is additional migration coverage, not a feature the old Python client supported and lost.

## Applicability (source-audited)

| Dimension | Scope |
|---|---|
| Models/connectors | All shipped Java source, including Tabular/Relational APIs and SQLite, DuckDB, PostgreSQL, Databricks SQL/Spark, Snowflake and file/S3 adapters. A PostgreSQL connection to Lakebase is covered as Java source, not live database validation. |
| Language | Java only for the added lane; existing Python/Rust lanes remain. External JDBC/model services are not analyzed merely because Java calls them. |
| Sampling | None directly: analyze Java host/provider code used around both in-memory and push-down operations; actual shared engine logic remains in the existing Rust analysis lane. |

Successful analysis is security-tool coverage, not connector compatibility or model E2E evidence.

## Purpose and history

Main `86c4f2129` scanned Python and its old C++ engine.
The migrated workflow correctly scans Python and Rust, but the newly added Java adapters also handle connections, requests and responses.
Without Java analysis, potential Java-specific vulnerabilities miss this check.
The policy is in `.github/workflows/codeql.yml`; it is developer tooling, not a documented prediction API or notebook feature.

## Small design

Add Java coverage to the existing workflow using the pinned action's supported extraction/build mode.
Verify that the chosen mode actually extracts `java/src/main/java`, including dependency resolution; do not treat a green job with no meaningful sources as coverage.
Use the existing Java build setup if a build is necessary, without loading native model code or adding Python callbacks.
Keep analysis permissions minimal and preserve current separation between analysis execution and release gates unless explicitly changed.
Rust, Python and Java public APIs are unchanged.

## Tests and acceptance

Validate workflow configuration and run the Java extraction/analysis job on a branch.
Inspect its source coverage and uploaded results; compilation alone is not proof of analysis.
Confirm existing Python/Rust lanes still run and review findings rather than suppressing them wholesale.
No model endpoint or GPU is needed.

Evidence: `.github/workflows/codeql.yml:51` and `java/src/main/java/com/nvidia/sdm/`.

## Implemented and verified

Java uses manual extraction around `mvn -f java/pom.xml clean compile`, followed by the existing security-extended analysis.
This resolves the real Maven dependencies, including optional adapters, without tests, JNI loading or model calls.
[GitHub's build-mode guidance](https://docs.github.com/en/code-security/reference/code-scanning/codeql/build-options-for-compiled-languages) supports explicit traced builds; a clean compile avoids incremental build omissions.
A small post-analysis check requires the source archive to contain the exact bytes of every shipped Java source.

| Evidence | Result |
|---|---|
| Local negative/ordering regressions | Six passed; reject missing, stale and duplicate sources and an empty source tree |
| Ruff / targeted mypy / diff whitespace | Passed |
| [Actual branch analysis](https://github.com/NVIDIA/structured-data-models-client/actions/runs/41751430654), code `3a16881af` | Java compiled all 101 sources; analysis and upload succeeded; archive guard verified all 101 |
| Uploaded Java analysis | `2091068805`, one finding reviewed below |
| Independent review | Wheel-provenance agent reviewed ownership, workflow sequencing and coverage checks; no material findings |

The Java finding (`java/comparison-with-wider-type`, alert #8) points to the integer loop versus long-returning `Table.getRowCount()` in `TimeSeriesArrow`.
Inspection of the pinned Arrow 19 bytecode confirms that `BaseTable` stores an integer row count and `getRowCount()` merely widens it; ordinary Table constructors also accept integer counts.
No supported-input overflow defect was established, and no product code or alert suppression was added.
An arbitrary custom subclass overriding that invariant was not treated as a normal Arrow input contract.
The entire branch workflow completed successfully, including Python and Rust.
Rust analysis `2091129751` reported zero results/errors; existing Python findings concern test-only workflow credential assertions, not a newly established product vulnerability.
Extraction coverage and successful analysis do not certify absence of vulnerabilities or connector/model E2E parity.
