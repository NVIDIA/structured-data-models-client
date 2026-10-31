<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Tabular inputs: use existing table types

**Recommendation: keep pandas, support Arrow directly, and avoid a new public `TabularData` class unless a concrete gap requires one.**
The Java implementation uses the API below; direct Python Arrow input remains a separate extension.

## Python: Arrow-backed pandas is already an efficient option

```python
frame = pd.read_csv("data.csv", engine="pyarrow", dtype_backend="pyarrow")
```

`engine` selects the CSV parser; `dtype_backend` selects how the resulting columns are stored.
These are reader options, not a universal DataFrame execution engine.
Existing frames can use `convert_dtypes(dtype_backend="pyarrow")`, which may allocate during conversion. [pandas Arrow guide](https://pandas.pydata.org/pandas-docs/stable/user_guide/pyarrow.html).

**Can this be as fast as direct Arrow? Potentially for data handoff; not proven for the full client.**
A local probe with pandas 3.0.6 / PyArrow 26.0.0 confirmed that `pa.array(series)` shared buffers for Arrow-backed integer, float and string columns.
Our simple direct helper accepted the numeric columns but required a different native-observation path for the Arrow string dtype; this is not evidence that public string input is unsupported.
Validation, sampling, casts and encoding can still allocate or perform extra work.

Keep pandas as the convenient input, including Arrow-backed columns.
Add direct `pyarrow.Table` / `RecordBatch` support for callers who already have Arrow, without converting through pandas.
Measure equivalent dtypes, nulls and rows through request preparation before claiming either route is faster.
Direct Arrow support is a separate extension, not currently available publicly.

## Java: there is an existing Arrow Table

Java's standard library has no dataframe type.
Apache Arrow provides `org.apache.arrow.vector.table.Table`, already in our dependency family.
It offers typed columns, dictionaries and row access; it is not the SDM `Table` class, which describes schema metadata.

**Expose one public input type: Arrow Java `Table`.**
Do not add public `VectorSchemaRoot`, dictionary-provider or `ArrowReader` overloads in this delivery, and do not introduce `TabularData`.
Those are lower-level Arrow building blocks: a root holds a batch, dictionaries interpret encoded values, and a reader advances through batches.
Users of our prediction API should not have to choose between them.
A future streaming/import API is separate work, justified by a concrete use case.

## Match the Python workflow

Both languages supply examples, rows to predict, target and task; then select the model at prediction time.
Keep Request model-neutral, rather than naming its factory `tabular`.

Existing Python:

```python
request = Request(context, predict, target="price", task="regression")
result = client.predict(request, model=KumoTabular())
```

Java, with Arrow Tables instead of DataFrames:

```java
Request request = new Request(context, predict, "price", "regression");
PredictionResult result = client.predict(request, "kumo-tabular");
```

Its four inputs match Python; optional settings can use Java options without changing their meaning.
PQL and fitted-session workflows should likewise retain `materialize`, `fit`, `predict` and `close` concepts.

Arrow Java `Table` is experimental in our pinned version 19.0.0.
Its constructor can share vectors, and `copy()` creates a deep copy; verify the pinned implementation before committing this public dependency. [Arrow Table API](https://arrow.apache.org/java/current/reference/org.apache.arrow.vector/org/apache/arrow/vector/table/Table.html).

**Lifetime rule, verified against Arrow 19:** Request and `Schema.fromData` take an owned snapshot when constructed.
The caller can then mutate or close the original tables without changing the request.
Arrow 19's public `exportTable` consumes its vectors, so borrowing it through that method would corrupt reusable inputs.
The adapter instead copies columns and writes an internal Arrow IPC snapshot, preserving dictionaries; there is no new public holder or closeable Request.
Repeated predictions reuse that snapshot, although decoding it for each core call still allocates.
This is a correctness-first lifetime adjustment, not a zero-copy claim; measure construction time and memory separately from repeated prediction.
A fitted session retains only required metadata after successful fit and needs no original context buffers.

End-to-end tests verify caller-table reuse/closure and dictionary values against the pinned implementation.
Keep batch/dictionary handling inside the adapter; reader-import conveniences are outside this delivery.
If this exposes a real ownership gap, keep storage management internal where possible rather than creating another dataframe API.

## Shared behavior and delivery

Python and Java send typed Arrow columns into Rust; Rust owns validation, sampling and model encoding, while the host owns input adaptation and HTTP.
Do not fabricate pandas conversion facts for native Arrow inputs.
Preserve null versus NaN, numeric precision, timestamps and dictionaries; Arrow row identity defaults to position, with entity keys explicitly named.
Keep existing pandas mutation/index behavior unchanged and review pandas index metadata before accepting exported Arrow tables as model features.

Java supports direct prediction, PQL with entity keys or supplied Arrow rows, and fitted sessions.
Closing the client attempts to close every owned session; a failed explicit session close can be retried.
Java Tabular can run beside Relational sampling with at most two packages active and coordinated shared-contract edits.
Compare correctness, preparation time and peak memory; internal Arrow support alone does not prove public API coverage or zero-copy execution.
