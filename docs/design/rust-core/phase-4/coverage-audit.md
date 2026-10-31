<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Phase 4 coverage and routing audit

Status: investigating from baseline `cb64d1cca`; previous completion claims are inputs to verify, not proof.

## Plan

1. Inventory supported prediction semantics, model inputs, connectors and public APIs from code.
2. Trace Python and Java entrypoints through preparation, prediction and fitted sessions to Rust, including refusal and fallback paths.
3. Build a matrix across prediction type × connector/input × model × language, separating support from actual test evidence.
4. Run focused public API journeys and legacy-entrypoint tripwires; use existing live resources where useful and record inaccessible coverage honestly.
5. Reconcile findings against the original Phase 4 scope and accepted decisions, obtain independent review, and publish the remaining-work list.

## Scope and evidence rules

- Include node regression, binary/multiclass classification, temporal aggregates, link prediction, ranking and forecasting where the model/API supports them.
- Include in-memory, SQLite, DuckDB, PostgreSQL, Databricks/Lakebase, Snowflake and other shipped source types found in the inventory.
- Java Snowflake is an explicit follow-up, not an accidental Phase 4 omission.
- Tabular source reads and model materialization are separate: do not imply `Schema.from_source` is supported for Tabular.
- Compare equivalent API capabilities and lifecycle behavior, without requiring identical language-specific types or spelling.
- Distinguish Python/Java host I/O and conversion from legacy Python sampling, model-policy and result-processing code.
- For each supported combination, record the default route, fallback/refusal behavior, source anchors and tested evidence.
- A shared Rust implementation is code evidence, not proof that every provider/language combination ran successfully.
- Preserve public APIs, approved boundaries and warehouse SQL; this audit does not silently expand product scope or remove legacy packages.

## Deliverables

A short human-readable support matrix and API-parity table in this document, with detailed machine-readable cases/evidence alongside it if needed.
A prioritized list of actual implementation gaps, active legacy paths, test gaps and explicitly deferred items.
Reproducible tests/logs for route claims, with credentials and user data excluded.
