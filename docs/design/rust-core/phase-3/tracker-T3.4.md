# T3.4: Databricks and contract v28

Lane a, from the phase head b7463a07 (T3.2, T3.3 with T3.19, T3.8, T3.14, T3.15 and T3.18 landed), under the structure design's T3.4 section (fixes C, E and F).
The work first pushed as `rust-core-phase-3-topic-T3.4-b` (b71baab1) is rebased here as a fresh branch: the test author's commits first, then T3.4's own tests, then the code.

## What changed

- **Contract v28** (design choice 3; T3.8's `ProviderChoice::Sql.namespace` and `DescribePart::Namespace` kept, T3.4's own variant of that name dropped):
  - `Relation.columns` and `Relation.filter`, opaque SQL text the core never reads, with the capability `derived_relations`; a derived relation for a provider without it is refused by name (rule `derived_relation`), through the shared `FROM` rendering;
  - the capabilities `sample_rows` and `neighbor_lookback`, `SampleEntities.rows` and `SamplerMemo.rows`, for the old client's `TABLESAMPLE` percentage and `_by_fkey` look-back;
  - `TableKey.unique` (P2-T2.2-3), counted as the reference counts a key the source keeps unique, in the shared key counts of every dialect;
  - `ProviderError.refuses_expression` (overnight decision 4), set by the host from the SQLSTATE class 22 or 42, for T3.5's view columns;
  - the data layer's built-in-only flags in `ProviderCapabilities` (P2-MID2-2), the `DataLayer` methods and dialect hooks gone;
  - `CatalogColumn.canonical`, off the wire; FX-6, the binding's test messages reading the crate's version.
- **One namespace model** (fix C): each dialect declares its namespace depth (DuckDB and SQLite 0, Postgres 1, Databricks 2) and how its source compares names, replacing T3.18's boolean; a namespace of more parts is refused by name.
- **The `databricks` dialect** (fix E): one `SqlDialect`, only what is Databricks's own: backtick quoting, three-part relations, the catalog statements with the old client's `lower(?)` parameters, key sets through `posexplode(from_json(?, ...))`, `TABLESAMPLE (p PERCENT) REPEATABLE (seed)` with the percentage as the reference's `str(float)` writes it (KO3-41), `QUALIFY ROW_NUMBER()` per seed, `_get_num_rows`'s count, a derived relation as `_quoted_source_name` reads a metric view's table; its catalog reader types each column as `_to_dtype` does and refuses a metric view read as a table.
- **The core**: a table's columns read before its keys, under the catalog's canonical names after; a declared foreign key's table found by the relation's dotted name; a declared unsupported type refused by name (P3B-1); for a provider declaring `sample_rows` or `neighbor_lookback`, every table's rows counted once for a materializer, the sample naming its rows, and each neighbor read by a foreign key looking back as `_by_fkey` does.
- **The host** (fix F): `Source.kind` and `Source.namespace_parts()` in `sdm_connectors`; one `providers.for_source` keyed on them; `_routing` keyed on kind, the `databricks-sql` and `databricks-spark` routes core-routed, defaults `old`; `DatabricksSqlProvider` and `DatabricksSparkProvider`, one implementation, the key set as one JSON text written as `_dumps` writes it, chunked as `_chunk_rows` chunks it; a Databricks failure's SQLSTATE read as `sql_state` reads it.

## Deviations and what is left

- SQL text that differs from the old client's where the result is the same, for review against the recorded statements: the key set's columns named by the core's seed names (`batch`, `key`, `time`, `start`, `end`) where the reference writes `__KUMO_BATCH__`, `__KUMO_ID__`, `e` and `s`; a lookup's duplicates removed in SQL where the reference removes them after the fetch; no outer window bounds (the reference binds them for data skipping, which the live timing run measures).
- Carried rows left for the packages now reworking their areas: P2-T2.8c-2 (answered parts, the host's results: T3.16), P2-MID2-3 (the unread-columns warning's fields: T3.17's wording), P2-MID2-4 (typed option values: T3.16's plain-data inputs), P2-T2.2-1 (a lookup's seed meanings as request fields, which rewrites every dialect's lookup goldens).
- The live recording of Databricks' core route (structure design step 3) waits for the owner's renewed login and the write guard.
