# T3.3 tracker: the warehouse harness

Lane H, stack 1. Brief: the orchestrator's `phase-3/briefs/T3.3.md`; design row T3.3, choices 1 and 2.

## Order of work

1. The warehouse loader and the capture writer, for the review round before C-1 (this section's first push).
2. The C-1 and E-0 job templates, the live write guard and the carried CI rows.
   The offline stand-in, its binary, its client, `handoff.py`'s warehouse legs on it and the per-leg CI jobs were built in rounds 1 to 3 and dropped in fix round 3 (owner decision B-lite, below).
3. C-1's captures, committed under the protected-path trailer when the orchestrator hands them over.

## The loader (`contract/runners/databases.py`)

- Each engine's column of the type map, in `TYPE_MAP` and `contract/type-map-vectors.json` together (`test_contract_loader.py` holds them equal), as the test author's interface names it (`ENGINE`, `column_type`, `stored`).
  An unsigned integer is declared in a type holding its range (as on Postgres); a type an engine has no column type for is stored in a form of its own and listed by `non_native` (KO2-47): on Databricks a time of day, an interval and a union; on Snowflake an interval, and the nested types as JSON (`ARRAY`, `OBJECT`, `VARIANT`).
- Design deviation, accepted by the orchestrator: Snowflake stores a time of day as `TIME(f)`: the engine has the type and the old client reads it (`_to_dtype`: `TIME` is `Dtype.time`), where the design's choice 1 says neither warehouse has one in tables.
- Zoned timestamps are `TIMESTAMP_LTZ(f)` on Snowflake and `TIMESTAMP` on Databricks; plain ones `TIMESTAMP_NTZ`; a nanosecond Databricks cannot hold is refused by name.
- Databricks keeps primary and foreign keys only: a `UNIQUE`, or a foreign key not referencing a primary key, is left out and returned as a note (`load`'s return value; the recording mode does not keep it yet, so the capture's `loader` stays empty).
- Every statement names its table fully qualified, so the live write guard (KO3-51) passes it; values are literals the engine reads back exactly (text and bytes as hexadecimal).
- The Postgres loader qualifies every name by its schema, so a case table named `pg_roles` is the case's, not `pg_catalog`'s.
- `load(duck, <leg>, target)` takes the recording mode's `WarehouseTarget` (T3.2): the leg's test schema holds one case's tables at a time across the run (`CaseSchema`, dropping only what it loaded), and a view given by its YAML is created from it.

## The capture writer (`contract/runners/captures.py`)

- Format `sdm-capture/1`, as the test author's interface names it (`write(path, capture)`, `read(path)`): a directory per case and leg, `capture.json` and each distinct Arrow table once under `tables/<sha256>.arrows`; each leg's `manifest.json` (`write_manifest`) in the shape T3.2's `capture_check.py` reads.
- Placeholders: each `SDM_CAPTURE_<NAME>` variable's value becomes `{<name>}`, the variables the recording mode resolves a case's placeholders from (T3.2), and hosts and users by their drivers' variables; in each case form a warehouse folds a name to, in every text, the tables' cells, field names and metadata included.
- The write fails, writing nothing, on any value of `PRIVATE` or of any `SDM_PRIVATE_*`, `SDM_LIVE_*` or `SDM_CAPTURE_*` variable, named by the variable, never the value.
- `captures.py import` turns `run_python.py --record`'s lines (T3.2's `sdm-record/1`) into capture directories, the placeholders and the search applied again.
- The converter: `CONVERTERS` reads every older version into the current one; the version-1 fixture is the test author's `contract/tests/t33-capture-v1`, recorded once by their `t33_capture_v1_record.py`.

## Probes before the first push

- The loader over every case of `contract/cases`: each case's tables are written for both engines or refused by name (`test_every_s1_case_is_written_or_named_on_each_warehouse`, and the test author's), and every statement passes the live write guard's check (`tests/test_t33_write_guard.py` runs the loader's statements through it).
- Outside JSON: a capture nested 100,000 levels is refused by name (`not a capture (RecursionError)`); an object 200,000 keys wide reads; a reference line nested past Python's recursion is refused by name at write time.
- The private-name check, seeded: a name left in the reference line, a cell, a field name, a statement, a binary cell, a URL's host and a token each fail the write with no file written (`contract/tests/test_captures.py`).

## The live runs, the guard and the carried rows

- `sdm-contract-run` reads the cases of the database it runs on (`case::load_for`): a warehouse-only case never runs on DuckDB.
- C-1 and E-0: `scripts/live/capture_run.py` (`plan`, the dry mode; `run`), `capture.job.template.json` and its notebook `capture_job.py` for the Spark and Lakebase legs; the two Databricks legs run one after the other.
  The recording's lines become captures (`captures.py import`), against which the core's SQL is compared with the old client's recorded SQL.
- The live write guard, `scripts/live/write_guard.py` (KO3-51), in the repository after the copy outside it was lost: every live write, DDL included, goes only to the one test schema of its warehouse.
  An allowlist (the owner's decision after round 4): one statement per text, with no comment, backslash, flow or pipe operator, escaped string or quoted name holding its own quote; the old client's reads (`SELECT` and `WITH` with no write verb, `INTO` or `SYSTEM$` function but `SYSTEM$READ_YAML_FROM_SEMANTIC_VIEW`, and the exact `SHOW`, `DESCRIBE`, `USE` and `ALTER SESSION SET` forms); and the loader's writes (`CREATE TABLE`, `INSERT INTO`, `DROP TABLE | VIEW | SEMANTIC VIEW | TYPE` without `CASCADE`, the metric view, `CREATE TYPE`, `COPY ... FROM STDIN`, and the semantic view's `CALL` with its YAML's name and base tables checked), each target in the test schema; everything else refused.
  One guard for every leg and path: `install` wraps the drivers' own classes (Snowflake's cursor and connection, Snowpark's session, Databricks SQL's cursor, Spark's session, classic and Connect, psycopg's cursors and connections, with every subclass), so a returned cursor or a session's own connection is guarded too; Spark's DataFrame and catalog writers and Snowflake's `execute_stream` are refused outright.
  The test schema comes from `SDM_LIVE_<DIALECT>_TEST_*`, else `SDM_LIVE_TEST_*`, and must be the one the cases resolve to (`SDM_CAPTURE_*`); no resource name is in the repository.
  Cost, named: the cases whose column names hold their own quote or a line break (`name.backtick-*`, `name.line-break-*`) are refused by the guard, so never recorded live.
  `capture_run.py` loads it from beside itself; a run without it, or without a test schema, does not start.
- Carried rows closed: P2-T2.7-1 (`layer.rs`), P2-T2.7-2 (`memory.rs`), P2-T2.14c-1, P2-T2.1a-3, P2-T2.16a-1, P2-T2.11-4, C28, OF-7's predicate, M5, C16's rest, P2-T2.16-14 and P2-T2.1o-1 (the slowest job is now `tooling`, about 43 minutes; `test`, `contract-python-ref-pandas3` and `engine` back at 60).
  ENV-2: `test_contract_loader.py`'s Postgres statement-capture tests pass in lane H's environment; closed.
  ENV-3: the prepush replay self-test passed on each of this package's pushes; closed.

## Fix round 1 (reviews of 5bfb59e2)

Codex: nested field metadata searched and kept, nested types kept through write and read, every read of a statement kept, empty nested columns loaded, named parameters kept.
Claude: semantic views created and dropped under the case's name, a leg's schema holding only the case (refused by name otherwise), placeholders replacing whole names only.

## Fix round 2 (reviews of 6f8ac662)

The stand-in checks a request's recorded tables and draw before exact replay, loads tables and draws of no rows, and serves Snowflake's unseeded draw as recorded; Lakebase is served on the local Postgres; the stand-in is built once in CI (`standin-binary`).
A Snowflake case creating a semantic view from its YAML has its tables created as the YAML names them (unquoted, so upper case), and C-1's Snowflake plan creates that view first.
From here the package is stacked on T3.2's candidate, on `rust-core-phase-3-topic-T3.3`.

## Fix round 3 (reviews of e8cdee80)

Owner decision B-lite: the offline stand-in is dropped (`standin.rs`, its capture reader, `sdm-standin`, `standin_client.py`, the `standin-binary` and per-leg CI jobs, `handoff.py`'s warehouse legs on it); the C-1 recording and the comparison of the core's SQL with the old client's recorded SQL stay.
A Snowflake semantic-view case keeps its table names, its columns created upper-cased and its YAML's base tables quoted, so the recording's read-back resolves.
C-1 and E-0 run the recording in process with every write through the repository's live write guard, refused outside the one test schema before the driver sees it; the jobs run the reference checkout's client, set `SDM_LIVE_CONFIG`, `SDM_CAPTURE_*` and the test schema from their parameters, and the Spark leg reads the notebook's session.

## Fix round 4 (reviews of 392f665a)

The guard became an allowlist enforced on the drivers' own classes for every leg (Lakebase's separate check removed, the loader's Lakebase drops without `CASCADE`); every bypass both reviews found is a test that is refused.
The comparator judges warehouse legs by every rule against the recorded reference lines, with no stand-in verdict or live-only items; the stand-in's policy section, gate runner and runner hook are removed.

## Fix round 5 (reviews of cdebf606)

Tests first: the round's refusal tests are committed before the code, and fail on cdebf606.
Reads became an allowlist too: a query, or what an `INSERT` fills its target with, calls only the closed list of pure functions the old client's generated SQL and the loader's literals use, by plain name; a quoted, qualified or unlisted function and any sequence word are refused, and every statement the old client recorded on Postgres passes.
The guard covers Snowflake's cursor base (so `DictCursor`) and its async connector, refuses Spark SQL given arguments to format, refuses Spark's and Delta's writer classes whole, admits only SQL commands at Spark Connect's command entry point, and guards subclasses defined after the install; checked against the real drivers (Snowflake connector 4.8.0, Databricks SQL 4.7.0, PySpark 4.0.1, Delta 4.4.1, Snowpark, psycopg).
The refusal census's real-data mode lost its stand-in warehouse legs.

## Fix round 6 (reviews of 6da8e532)

Tests first again.
A table's and a composite type's columns are held to the function rule a read is (no `CHECK`, `GENERATED` or `DEFAULT`, no call but a type's parameters), a name followed by `(` after `AS` or `::` must be a type or an alias's list of names, and Snowflake's `cmd_query` is guarded.

## Fix round 7 (reviews of 0c897542)

Tests first again.
A table's columns are each a name and a built-in type the loader writes (on Postgres also a composite type in the test schema) or a key constraint referencing the test schema; `LIKE`, a domain or another schema's type, `INHERITS`, `PARTITION OF` and `OF` are refused, and inside a cast `AS` names a type, never an alias's column list.
Codex's round 7, tests first: no keyword or type exemption admits a qualified name or a call, names are plain ASCII outside strings, a cast names only a built-in type, Spark Connect is guarded at its execution boundary (every plan a read or a SQL command, every SQL text in it checked), and Spark's global temporary views are refused.

## Fix round 8 (re-grade of b7c36db7 under the corrected scope)

The guard is a safety net against accidental writes by our own code; crafted bypasses are P3, for the owner's restricted database roles.
Tests first: the recorder's own statements pass each leg's guard (its Lakebase zone read needed `current_setting`), each leg's plan leaves out by name the ten cases whose names hold a backtick or a line break (`capture_run.NOT_LIVE`), and the run records case by case, a failed case written with its reason to `<leg>.failed.json` while the others are recorded and imported.

## Fix round 9 (reviews of 796406aa)

Tests first: a case fails, with none of it imported, when the guard refused any statement during it (the guard keeps every refusal, so one the old client's call swallows is still seen), when the runner fails (its message surfaced) or when any other error is raised; the other cases go on.

