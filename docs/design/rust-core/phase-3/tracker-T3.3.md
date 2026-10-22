# T3.3 tracker: the warehouse harness

Lane H, stack 1. Brief: the orchestrator's `phase-3/briefs/T3.3.md`; design row T3.3, choices 1 and 2.

## Order of work

1. The warehouse loader and the capture writer, for the review round before C-1 (this section's first push).
2. The stand-in (`sdm-testkit`'s `src/standin.rs`, its capture reader, the `sdm-standin` binary, `contract/runners/standin_client.py`), `handoff.py`'s warehouse legs, the C-1 and E-0 job templates, the per-leg CI jobs and the carried CI rows.
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

- The loader over every case of `contract/cases`: each case's tables are written for both engines or refused by name (`test_every_s1_case_is_written_or_named_on_each_warehouse`, and the test author's), and every statement passes the live write guard's check offline (run once from the guard outside the repository).
- Outside JSON: a capture nested 100,000 levels is refused by name (`not a capture (RecursionError)`); an object 200,000 keys wide reads; a reference line nested past Python's recursion is refused by name at write time.
- The private-name check, seeded: a name left in the reference line, a cell, a field name, a statement, a binary cell, a URL's host and a token each fail the write with no file written (`contract/tests/test_captures.py`).

## The stand-in, the harness and the carried rows

- The stand-in (`sdm-testkit`'s `standin.rs`, `sdm-standin`, `standin_client.py`): exact replay, then the DuckDB local engine with the capture's NULL order and zone, a draw served only as recorded, a missing table or draw refused by name, B1, B2 and B4; `standin_client.open` gives `run_python.py` the forwarding provider `provider(step, keys)`, and serves Lakebase on the local Postgres, the case's tables loaded under the capture's `{schema}`.
- `handoff.py` reads each case's legs as `run_python.py` does: an S2 case never runs on DuckDB, SQLite or Postgres; each touched case's warehouse legs run on python-core offline where the core reports the leg's provider, held to the capture's reference line (Lakebase's to the Postgres line until its capture exists); a leg with no reference is named pending live.
  Exit 5's test (a difference on Lakebase alone stops the stamp) passes once the core reports a Lakebase provider (T3.7); until then the leg is named, not judged.
- `sdm-contract-run` reads the cases of the database it runs on (`case::load_for`): a warehouse-only case never runs on DuckDB.
- The per-leg CI jobs, `contract-python (ref-pandas3, <leg>)` and `cross-host (<leg>)`, judge the cases with a reference on the leg where the core or rust-host reports its provider, and list the rest.
- C-1 and E-0: `scripts/live/capture_run.py` (`plan`, the dry mode; `run`), `capture.job.template.json` and its notebook `capture_job.py` for the Spark and Lakebase legs; the two Databricks legs run one after the other.
- Carried rows closed: P2-T2.7-1 (`layer.rs`), P2-T2.7-2 (`memory.rs`), P2-T2.14c-1, P2-T2.1a-3, P2-T2.16a-1, P2-T2.11-4, C28, OF-7's predicate, M5, C16's rest, P2-T2.16-14 and P2-T2.1o-1 (the slowest job is now `tooling`, about 43 minutes; `test`, `contract-python-ref-pandas3` and `engine` back at 60).
  ENV-2: `test_contract_loader.py`'s Postgres statement-capture tests pass in lane H's environment; closed.
  ENV-3: the prepush replay self-test passed on each of this package's pushes; closed.
- `handoff.py` builds the previous round's Rust tests in a target directory of their own: built into the caller's, their artifacts were newer than the candidate's sources and the next build took them as fresh.

## Fix round 1 (reviews of 5bfb59e2)

Codex: nested field metadata searched and kept, nested types kept through write and read, every read of a statement kept, empty nested columns loaded, named parameters kept.
Claude: semantic views created and dropped under the case's name, a leg's schema holding only the case (refused by name otherwise), placeholders replacing whole names only.

## Fix round 2 (reviews of 6f8ac662)

The stand-in checks a request's recorded tables and draw before exact replay, loads tables and draws of no rows, and serves Snowflake's unseeded draw as recorded; Lakebase is served on the local Postgres; the stand-in is built once in CI (`standin-binary`).
A Snowflake case creating a semantic view from its YAML has its tables created as the YAML names them (unquoted, so upper case), and C-1's Snowflake plan creates that view first.
From here the package is stacked on T3.2's candidate, on `rust-core-phase-3-topic-T3.3`.

## Fix round 3 (reviews of e8cdee80)

The stand-in serves a draw only for the request's own seed, and a recorded view read by exact replay.
A Snowflake semantic-view case keeps its table names, its columns created upper-cased and its YAML's base tables quoted, so the recording's read-back resolves.
C-1 and E-0 run the recording in process with every write through the live write guard, refused outside the one test schema; the jobs run the reference checkout's client and read their environment file.
The guard module itself was at `/mnt/localssd/tmp/live-setup/write_guard.py`, which is gone from this machine: C-1 needs it restored (T3.7 brings it into the repository).
