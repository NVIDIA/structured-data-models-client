# T3.4: Databricks and contract v28

Lane b, from the phase head d1e02d7c (T3.1 landed), stacked on T3.2's candidate 5c65296c; T3.3's candidate follows.
The acceptance tests are the test author's (`T3.4 tests:` commits from `rust-core-phase-3-topic-T3.4-tests2`), brought onto this lane with their subjects kept.

## What changed

- **Contract v28** (design choice 3), every new field optional with a default, so no existing message changes shape:
  - `Relation.columns` (each a name with a source column or an SQL text expression) and `Relation.filter`, opaque to the core; the capability `derived_relations`, without which a derived relation is refused by name (rule `derived_relation`, `refuse_derived`), held by the data layer's one rendering of a relation;
  - the capabilities `sample_rows` and `neighbor_lookback`, `SampleEntities.rows` and `SamplerMemo.rows`, for the old client's warehouse samplers' `TABLESAMPLE` percentage and `_by_fkey` look-back;
  - `TableKey.unique` (carried row P2-T2.2-3);
  - the describe part `namespace` (`SELECT current_catalog(), current_schema()`), and `CatalogColumn.canonical`, not on the wire;
  - `ProviderError.refuses_expression` (overnight decision 4), set by the host from the SQLSTATE class 22 or 42, for T3.5's view columns;
  - the binding's test messages read the version from the crate (FX-6); the v26 vectors of a derived relation are refused through v27 and read after it (`refused_through`); the vocabulary guard admits the two opaque SQL texts by name and nothing else.
  - T3.8 adds the provider choice's `namespace` in the same version (coordinated; whichever lands second rewrites the working entry, decision H4).
- **The `databricks` dialect** (`sdm-data/src/databricks.rs`): every request kind as the old client's Databricks engine writes it; the catalog statements its text and `lower(?)` parameters, so exact replay serves the recorded answers; key sets bound through `posexplode(from_json(?, ...))`, the position the batch; `TABLESAMPLE (p PERCENT) REPEATABLE (seed)` with the percentage the reference's arithmetic gives, written as its `str(float)` (KO3-41); `QUALIFY ROW_NUMBER()` per seed; a derived relation read as `_quoted_source_name` reads a metric view's table; the catalog reader typing each column by its declaration as `_to_dtype` does, refusing a metric view read as a table by name, single-column keys and foreign keys only.
  The key set reaches the provider as `bound_keys` writes it: the batch left out, the key coerced as `_coerce_id` coerces it, each seed time its wall clock to the second.
- **The core** (`sdm-core`): a table's columns read before its keys, the catalog's canonical names adopted for every later read (DB/table.py:262); a declared foreign key's table found by the relation's parts joined by dots; each key the source keeps unique counted as the reference counts one; a declaration the reference reads as unsupported refused by name as P1B-10 refuses a struct; for a provider declaring `sample_rows` or `neighbor_lookback`, every table's rows counted once for a materializer in the graph's order, the entity sample naming its rows, and each neighbor read by a foreign key over a timed table looking back as `_by_fkey` does.
- **The host** (`sdm_client._rust.providers.databricks`): `DatabricksSqlProvider` (`fetchall_arrow`) and `DatabricksSparkProvider` (`spark.sql(...).toArrow()` through `SparkConnection`), one implementation; scalars bound by the codec; a key set as one JSON parameter, chunked and its batch shifted as `_chunk_rows` and `_flatten` do.
  Backend detection tells a SQL warehouse, a Spark session and a metric view apart (`_route_of`, private); the `databricks-sql` and `databricks-spark` routes are core-routed, defaults `old`; a Databricks failure's SQLSTATE read as `sql_state` reads it, `refuses_expression` set for classes 22 and 42.
  The serving path (`client.connect/databricks-serving/any`) already sends the core's request through the serving transport.

## Deviations and what is left

- SQL text differs from the old client's where the result is the same (H9 is diagnostic on stand-in legs, KO3-45): the batch alias written lower case as the contract names it; a lookup's duplicates removed in SQL (`QUALIFY ROW_NUMBER() ... = 1`, `COUNT(*) OVER` for the matches) where the reference removes them after the fetch; the window's outer bounds read from the seeds by scalar subqueries where the reference binds them, which E-4 times live; the seed times named by the core's seed columns.
- Waiting on T3.8: the provider choice's `namespace` (the golden `the_catalog_statements_are_the_old_client_s_with_its_parameters` parses a choice naming one), and the core's read of the current namespace where a part is null.
- Waiting on T3.3: the stand-in extended for v28's relation fields, and the cases' Databricks legs run on it.
- Waiting on T3.2's fix 1: `Standin.plugged` keeping the choice's namespace.

## Pending-live and live-only cases

Filled in once the cases run on the stand-in.
