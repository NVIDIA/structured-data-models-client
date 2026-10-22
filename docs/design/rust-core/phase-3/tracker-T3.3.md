# T3.3 tracker: the warehouse harness

Lane H, stack 1. Brief: the orchestrator's `phase-3/briefs/T3.3.md`; design row T3.3, choices 1 and 2.

## Order of work

1. The warehouse loader and the capture writer, for the review round before C-1 (this section's first push).
2. The stand-in (`sdm-testkit`'s `src/standin.rs`, its capture reader, the `sdm-standin` binary, `contract/runners/standin_client.py`), `handoff.py`'s warehouse legs, the C-1 and E-0 job templates, the per-leg CI jobs and the carried CI rows.
3. C-1's captures, committed under the protected-path trailer when the orchestrator hands them over.

## The loader (`contract/runners/databases.py`)

- Each engine's column of the type map, in `TYPE_MAP` and `contract/type-map-vectors.json` together (`test_contract_loader.py` holds them equal).
  A type an engine stores no column of says why in `<engine>_not_stored`; `unstored(duck, leg)` names each such column, so a case holding one does not run on that leg (KO2-47).
  Databricks stores no unsigned integer, time of day, interval or union; Snowflake no unsigned integer or interval.
- Snowflake stores a time of day as `TIME(f)`: the engine has the type and the old client reads it (`_to_dtype`: `TIME` is `Dtype.time`), where the design's choice 1 says neither warehouse has one in tables.
  This is a deviation from the design's wording, kept because the rule is "a type an engine cannot store is listed", and Snowflake can.
- Zoned timestamps are `TIMESTAMP_LTZ(f)` on Snowflake and `TIMESTAMP` on Databricks; plain ones `TIMESTAMP_NTZ`; a nanosecond Databricks cannot hold is refused by name.
- Databricks keeps primary and foreign keys only: a `UNIQUE`, or a foreign key not referencing a primary key, is left out and returned as a note, which the capture records (`loader`).
- Every statement names its table fully qualified, so the live write guard (KO3-51) passes it; values are literals the engine reads back exactly (text and bytes as hexadecimal, so no quote or backslash in a value can end a literal).
- The Postgres loader qualifies every name by its schema (the connection's current schema by default), so a case table named `pg_roles` is the case's, not `pg_catalog`'s; Lakebase loads into its one test schema from an open connection.
- `CaseSchema` keeps one leg's test schema holding exactly one case's tables across a run, dropping only what it loaded.

## The capture writer (`contract/runners/captures.py`)

- Format `sdm-capture/1`: one JSON file per case and leg under `contract/captures/<leg>/`, each distinct Arrow table once under `tables/<sha256>.arrows` (an uncompressed IPC stream, which `sdm-testkit` reads with no new dependency).
- Placeholders for every private name the recording knows, in each case form a warehouse folds it to, in every text: statements, parameters, the reference line, and the tables' cells, field names and metadata.
- The write fails, writing no file, on any value of the private environment variables (`PRIVATE_VARIABLES`, `SDM_PRIVATE_*`, `SDM_LIVE_*`, and a URL's host parts), named by the variable, never the value.
- The converter: `CONVERTERS` reads every older version into the current one; `captures.py convert DIR` rewrites a directory. The version-1 fixture is `contract/fixtures/captures/v1`.

## Probes before the first push

- The loader over every case of `contract/cases`: 309 load on Databricks and 311 on Snowflake; the rest are named by `unstored` (6 and 4: unsigned integers, a union, an interval, a time of day); every statement passes the live write guard's check offline.
- Outside JSON: a capture nested 100,000 levels is refused by name (`not a capture (RecursionError)`); an object 200,000 keys wide reads; a reference line nested past Python's recursion is refused by name at write time.
- The private-name check, seeded: a name left in the reference line, a cell, a field name, a statement, a binary cell, a URL's host and a token each fail the write with no file written (`contract/tests/test_captures.py`).
