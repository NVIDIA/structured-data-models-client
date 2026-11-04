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
