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
- The live write guard, `scripts/live/write_guard.py` (KO3-51), rewritten in the repository from the owner-approved spec after the copy outside it was lost: every live write, DDL included, goes only to the one test schema of its warehouse, fully qualified; reads pass; anything not recognised is refused.
  The test schema comes from `SDM_LIVE_<DIALECT>_TEST_DATABASE` and `SDM_LIVE_<DIALECT>_TEST_SCHEMA`, else `SDM_LIVE_TEST_DATABASE` and `SDM_LIVE_TEST_SCHEMA`; no resource name is in the repository.
  Difference from the spec, named: on Snowflake `CALL SYSTEM$CREATE_SEMANTIC_VIEW_FROM_YAML('<test schema>', ...)` is allowed (the loader creates a semantic view that way, in the schema the call names); every other CALL is refused.
  `capture_run.py` loads it from beside itself, never from a path given at run time; a run without it, or without a test schema, does not start.
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
