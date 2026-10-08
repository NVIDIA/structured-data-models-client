# Phase 5.5 acceptance contract

The acceptance checks in this directory test behavior through a small, fixed interface.
This file is that interface.
Everything it doesn't name is the implementer's choice.

The checks are locked: the implementer changes code, never these files.
`test_fix_*.py` run in CI as regression tests; `accept_*.py` compare with `BASE` (`f9951d7`) and run only in the phase gate.
Where a check needs today's behavior, it runs `BASE`'s `packages/*/src` (with this checkout's compiled engine extension) in a child process, through `_fixes.py`.

## Time windows (P55.tz-e2e, P55.tz-shape, P55.tz-spark)

- The public API is the interface: `Schema.from_source`, `PredictiveQuery.materialize`, `SDMClient.predict`, with the corpus' stand-in NIM (`scripts/parity/_parity/nim.py`, `SyntheticSession`).
- PostgreSQL and Databricks: every time bound in a window statement's parameters, inside the JSON key list and outside it, names the same instant as today's naive string and ends with `+00:00`, unless the time column is a string.
  Databricks' window SQL text stays exactly today's.
- Snowflake: for a `TIMESTAMP_LTZ` (`TIMESTAMP_TZ`) time column, each windowed statement reads every bound, the key list's and each outer `?`, through `TO_TIMESTAMP_LTZ(x, '<format>')` (`TO_TIMESTAMP_TZ`), with one format whose elements are among `YYYY MM DD HH24 MI SS FF TZH TZM` and double-quoted literals, including `TZH`.
  No `TIMESTAMP_NTZ` cast is left in those statements, every time string carries `+00:00`, and each matches the format.
  `TIMESTAMP_NTZ` and `DATE` columns keep today's statements exactly.
- A windowed statement is one whose SQL defines `__KUMO_END_TIME__`.
- `test_fix_spark_local.py` is opt in: `SDM_SPARK_LOCAL=1`, pyspark 4.2 or later (it needs `QUALIFY`), and a JDK in `JAVA_HOME`.

## Scan caps (P55.caps)

- PostgreSQL: `PostgresTable._SAMPLE_PREFILTER_MIN_BLOCKS` (16,384) and `PostgresTable._SAMPLE_PREFILTER_TARGET_BLOCKS` (2,048) are class attributes, read when a sample is taken, so a check can shrink them.
- A prefiltered PostgreSQL sample is today's statement with ` TABLESAMPLE SYSTEM (<repr(100 * f)>) REPEATABLE (0)` right after the table's quoted name, byte for byte otherwise.
- The size probe is one statement per unkeyed table, mentioning `pg_relation_size` or `relpages`, sent before its inference sample and recorded as the step right before it.
- Snowflake: the prefilter is `SAMPLE SYSTEM (<repr(100 * f)>) SEED (<s>)` right after the table's name, `s` 0 for the inference sample and the draw's seed modulo 2^31 for a seeded entity draw; whitespace around the clause is free.
  The listing is read from `SHOW TABLES` by field name (`rows`, `bytes`, `kind`, `is_external`, `is_hybrid`, `is_iceberg`, `is_dynamic`).
- Databricks: the size of a table comes from `DESCRIBE DETAIL <quoted name>` (`sizeInBytes`).
  Above the threshold the inference sample is `SELECT <today's select list> FROM (SELECT <cols> FROM <quoted name> LIMIT <M>)[ alias] <today's ORDER BY and LIMIT 1000>`, `M` between 10,000 and 1,000,000, `<cols>` `*` or every column the select list names.
  The source and expression samples of one table use the same subquery text.
  A table under the threshold, or whose size can't be read, sends today's statements, apart from metadata reads (`DESCRIBE`, `SHOW`, `ANALYZE`, and `information_schema.tables` queries naming `table_type`).
- Tests over Databricks and Snowflake use the stub drivers of `_warehouse_stub.py`.
  A metadata statement the stub doesn't know fails with the driver's own error, as an unsupported statement would.

## Composite links (P55.link)

- The interface is `Graph.from_duckdb`, `from_sqlite`, `from_postgres`, `from_databricks`, `from_snowflake`, a composite `primary_key`, and `Graph.link`.
- Linking sends exactly one statement naming the parent's quoted source name that isn't an inference sample: the lookup.
  It ends with `LIMIT 1`, holds the parent's derived key expression as its column holds it (`table[key].expr`), has no `TABLESAMPLE` or `SAMPLE`, and takes one parameter: a JSON array, as text, of the values, each a string or a one-key object or one-element array holding one.
  The values are sorted, distinct, at most 1,000, and are the folds of the rows of the child's inference sample with no null part.
- A null part in the child's reference is still refused with a message mentioning `null`, before or after the lookup.
- `qa/composite_link_matrix.py` is the opt-in A4 script.

## PostgreSQL reads (P55.pg-read)

- `PostgresSource._run` works on a connection whose `row_factory` is `dict_row` or `namedtuple_row`, and leaves the connection's `row_factory` as it was.
- It returns a `timetz` column as `string`, the value PostgreSQL writes, such as `12:00:00+05:30`.
- `sdm_connectors.sql.rows_to_arrow` raises an exception whose message names the column when a value doesn't fit the column's declared type; a column without a declared type still holds mixed values as text.

## Recordings (P55.tz-corpus)

- The scenarios `postgres/count#timestamptz-new-york`, `postgres/inactivity#timestamptz-kolkata`, `postgres/count#timestamp-kolkata`, `duckdb/count#timestamptz-new-york` and `duckdb/count#timestamp-new-york` are recorded.
- Each records `data.session`, whose JSON names its zone, `data.column_types`, which spells a zoned type `timestamptz`, `... with time zone`, `TIMESTAMP_LTZ` or `TIMESTAMP_TZ` and a naive one any other way (`timestamp`, `... without time zone`, `TIMESTAMP_NTZ`), and `twin`, top level or in `data`, naming its twin.
- A twin has the same backend, task kind and calls, and the same `data` apart from `session`, `column_types` and `twin`; its session is absent or UTC, its column types absent or naive.
- `FORMAT.md` names the keys in backticks, and `tests/parity/schema/transcript.schema.json` describes them.
- `python scripts/parity/run_corpus.py record --corpus DIR --only NAME` records one scenario into `DIR`, also with `BASE`'s packages on `PYTHONPATH`.
- `python scripts/parity/check_only_offset.py OLD NEW [BACKEND ...]` exits 0 when every transcript of `NEW` that `OLD` also holds differs from it only by `+00:00` on bound-shaped strings in step parameters (inside JSON text too) and by inserted size-probe steps, comparing Arrow files by content after renumbering.
  A scenario only `NEW` holds doesn't fail it.
  Any other difference exits non-zero, without a traceback.
