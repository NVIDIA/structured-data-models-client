# Phase 3 tracker: T3.2

The design branch owns `tracker.md`; this file holds T3.2's section until it folds in.
T3.1, the previous package on lane A, landed at d1e02d7c.

## Part 1: the recording mode, its completeness check and the case format (reviewed before C-1)

### The case format (both loaders)

- `warehouses`: the warehouse legs an S2 case runs on, and only there; at least one, none twice; only on an S2 case, which must name them; never beside `databases`.
- An S1 case without `databases` runs on every warehouse leg as well, except the two 100,001-row batching cases (KO3-45); `run_python.py`'s `case_legs` and `case.rs`'s `Case::warehouses` say the same.
- `input.view`: `kind` (`metric_view` on the Databricks legs, `semantic_view` on `snowflake`), `name`, optional `yaml` and `trust_expressions`; only beside `warehouses`.
- `input.connect`: `warehouse`, `spark`, `snowpark` or `lakebase`; the case names exactly that path's leg.
- `materialize.num_context_samples`: passed as written, as the sampling options are.
- P2-T2.1i-1: both loaders refuse `NaN` and `Infinity` (not JSON) and read a `fail_call` of `-0` as 0.
- Placeholders (`{catalog}`, `{schema}`, an existing view's `{shop_metric_view}`) in `input.from_source`, `input.edits` and `input.view` are resolved on a warehouse leg from the live configuration; an unknown one fails the run by name.
- Each rule has accept and refuse vectors in `contract/tests/malformed-cases.json` where one path can write it, and the rest in `contract/tests/test_t32_recording.py`.

### The recording mode (`run_python.py --mode python-ref --database <leg> --record FILE`)

- The live configuration is a JSON file outside the repository, named by `SDM_LIVE_CONFIG` (`sdm-live/1`): per leg, the placeholder names and how the connection opens (`driver`, `warehouse`, `lakebase`), any value read from the environment as `{"env": NAME}`.
- Each case's tables are loaded by `databases.load(duck, leg, WarehouseTarget(connection, names, view))`, T3.3's warehouse loader, which also creates a view from its YAML.
- The capture wraps every driver call the old client makes on each leg, per connection, never by patching a driver class: Databricks SQL (`execute`, `executemany`, `fetchall_arrow`, `fetchmany_arrow`, `fetchall`, `fetchone`, `fetchmany`), Spark (`spark.sql` and the frame's `toArrow`, `collect`, `first`, `toPandas`), Snowflake (`execute` under `qmark`, `fetch_arrow_all`, `fetch_pandas_all`) and psycopg (`execute`, `executemany`, `copy`, `stream`, P2-T2.1a-2, now the Postgres leg's capture too).
- A read the driver lacks is never offered, since the old client's reader picks the first Arrow read the cursor has.
- One `sdm-capture/1` line per case and leg: the reference line, every statement with its parameters and each result whole as Arrow IPC, every table as the warehouse returns it, each view-derived relation as the warehouse evaluates it (the read without its sample's `LIMIT`), every random draw (table, fraction and seed as written, and the rows it picked), the session zone, the versions, the recording and reference commits, and the case's input hash.
- Before anything is written, each private name becomes its placeholder (whole names, in any case), and the write fails, naming the case and never the value, on any private value left in the line, its tables included.

### The completeness check (`capture_check.py captures` and `captures-only`)

- `captures --cases ... --capture ... [--leg ...] [--locked]`: each line complete (its statements the reference line's own, each result held whole, the draws exactly those its statements made, every table of the case's data once, each derived relation evaluated or refused, the versions, the zone and the commits); every case and leg without a capture of its current input listed as pending live, a failure with `--locked`.
- `captures-only --final SHA [--commit SHA]`: the commits after the final candidate change only `contract/captures/**`, delete nothing, and every line they add or change names the final candidate as its recording commit.
- `contract/captures/**` is a protected path.

### Probes

- Outside JSON: a capture line and a case file are read by `json`, which refuses depth past its limit as `RecursionError`, caught and named by both readers.
- Seeded faults: `test_an_incomplete_capture_fails_by_name` drops or retypes each piece of evidence the completeness check reads (a draw, its fraction, its seed, a table, a statement, a result, its Arrow, its row count, a relation, the zone, the driver's version, the commit, the format, the reference line) and each fails by name.
