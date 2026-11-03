<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Replace a database connection

Python `schema.with_source(source)` and Java `schema.withSource(source)` return a new schema using the replacement connection.
The original schema, queries and requests remain unchanged.
Neither connection is closed or modified by rebinding.

```text
Python / Java: replacement Source + existing declarations
                         ↓
Rust: check source compatibility → refresh tables/types → retain declarations
                         ↓
Python / Java: new Schema → materialize → predict
```

Backend, namespace, source kind, semantic-model identity and expression-trust settings must match.
Changing database layout is a separate schema-construction operation.
Rust preserves keys, links, semantic types, time/end roles, calculated columns and inference choices.
It checks table/column availability and compatible types using existing source admission, without fresh inference or a full key-uniqueness scan.
In-memory schemas cannot be rebound.
Missing tables/columns and incompatible sources fail without changing either schema.

The old engine mutated table handles and could close its previous connection.
Returning a new schema improves ownership and prevents unrelated prepared work from switching databases.

Tests use separate SQLite/DuckDB connections with identically named tables but different values, assert independent expected labels, preserve declarations, exercise failures and confirm original queries still work.
Read-only warehouse tests reconnect, gather labels/materialize and predict against the existing model endpoint.

## Verification

Production native contract 66 passed 22 Python local tests covering rebinding and neighboring SQL-column behavior; one existing credentialed test was skipped in the local run.
The separate live Python run passed all five PostgreSQL, Databricks SQL, Snowflake, Databricks metric-view and Snowflake semantic-view reconnect journeys through the real relational model endpoint.
Namespace changes and changed semantic expression-trust settings were refused before source reads.
Java passed SQLite/DuckDB public workflows and a PostgreSQL reconnect → materialize → live prediction journey using the existing parity fixture/configuration helpers.
Both hosts cover reconnecting after the original connection has already closed.
An additional test in each host rejects a replacement DuckDB column whose declared type changed from numeric to text, without harming the original source.

A focused run against the pinned old engine confirmed that rebinding retained its key declaration and used replacement data, while mutating the existing table object.
The new API retains those useful capabilities and isolates the original schema and prepared payloads.
The Rust attachment inventory/schema-lock checks, core compilation, Python lint and targeted type checking passed.
