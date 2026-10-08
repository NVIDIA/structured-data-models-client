# Phase 2 acceptance contract

The acceptance checks in this directory test behavior through a small, fixed interface.
This file is that interface.
Everything it doesn't name is the implementer's choice, documented in `tests/parity/FORMAT.md`.

## Layout

- `tests/parity/FORMAT.md` documents transcript format v1.
- `tests/parity/schema/transcript.schema.json` is the JSON Schema every transcript validates against.
- Recorded transcripts live under `tests/parity/transcripts/`, one JSON file per scenario, with their Arrow IPC files beside them.
  Only recorded transcripts and the Arrow files they reference are stored there; a `pending-live` scenario has no file there until it's recorded.
- Arrow IPC files may use the IPC file or the IPC stream format.

## Transcript keys

A transcript is a JSON object with at least these keys:

| Key | Holds |
|---|---|
| `format_version` | The integer `1` |
| `name` | The scenario name, matching `^[A-Za-z0-9_./#-]+$` |
| `backend` | `sqlite`, `duckdb`, `postgres` or `memory` for offline scenarios; `databricks`, `lakebase` or `snowflake` for live ones; any other value for the tabular cases |
| `task_kind` | `count`, `classification`, `regression`, `forecast`, `ranking` or `inactivity` for relational scenarios |
| `data` | How the data is generated |
| `calls` | An array describing the public API calls and their arguments; the PQL text appears in it as a string |
| `steps` | The ordered database steps |
| `payloads` | An array of the normalized payloads, as parsed JSON objects |
| `result` | The decoded `PredictionResult.to_pandas()` result: a path, relative to the transcript's directory, of an Arrow IPC file |

Each step is an object with an `op` key, one of `execute`, `ingest`, `register`, `commit` and `paramstyle`.
An `execute` step also has:
- `sql`, the exact SQL text as a string;
- `params`, its parameters as JSON (`null` when there are none);
- `result`, a path relative to the transcript's directory of an Arrow IPC file, or `null`;
- `error`, `null` or a JSON object describing the error the call raised.

The only normalized payload field is `metadata.request_id`.

## `scripts/parity/run_corpus.py`

- `list --json` prints, on stdout, only a JSON array with one object per scenario: `name`, `backend`, `task_kind`, `status` (`recorded` or `pending-live`) and `transcript` (its path, relative to the repository root, or `null` while pending).
- `record [--corpus DIR] [--only NAME[,NAME...]]` records scenarios into `DIR` (default `tests/parity/transcripts`).
  Recording a live scenario without its credentials, whether it is `pending-live` or already recorded, exits non-zero, with a message rather than a traceback, and writes nothing.
- `replay [--corpus DIR] [--only NAME[,NAME...]]` replays every recorded scenario in `DIR`, or only the named ones.
  - It replays every selected scenario even after one fails.
  - For each scenario it prints one line, `PASS <name>` or `FAIL <name> <reason>`.
  - When the SQL text or parameters differ, the reason names the step as `step <k>`, where `k` is the 0-based index into `steps`.
  - It exits 0 only if every selected scenario passed.
  - It needs no network, no database server and no credentials.

## `scripts/parity/diff.py`

- `diff.py A B` compares two transcripts, or two payload sets (JSON arrays of payload objects).
  - For transcripts, it also compares the Arrow files that the transcripts reference.
  - Identical inputs: exit 0 and nothing on stdout.
  - Different inputs: exit 1, and the first line of stdout starts with the path of the first difference, followed by a space or the end of the line.
  - Paths use `.key`, `[index]` and `["key"]`, e.g. `[1].context.related_tables.orders.rows[7].price` for a payload set, or `steps[3].sql` for a transcript.
    A row's cell is named by its column name or by its column index.
  - Bad input: any other exit status.
- `diff.py --self-test` runs the tool's own seeded self-test and exits 0 only if it passes.

## `scripts/parity/scrub.py`

- `scrub.py scrub SRC DST` copies the transcripts and Arrow files under the directory `SRC` to `DST`, scrubbed, keeping their paths relative to the directory.
  Besides what it detects by pattern, it treats the values of these environment variables as secrets when they're set: `DATABRICKS_SERVER_HOSTNAME`, `DATABRICKS_TOKEN`, `DATABRICKS_HTTP_PATH`, `DATABRICKS_WAREHOUSE_ID`, `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_USER`, `SNOWFLAKE_PASSWORD`, `PGHOST`, `PGUSER`, `PGPASSWORD`.
- `scrub.py scan [PATH...]` scans JSON and Arrow files under each path (default `tests/parity/transcripts`), exits 0 if it finds no secret and 1 if it finds any, and lists what it found.

## `scripts/parity/coverage.py`

- `coverage.py [--out PATH]` runs the corpus under branch coverage and writes the report to `PATH` (default `tests/parity/coverage.json`).
- The report is a JSON object with a `crates` object, keyed by destination crate.
  Each crate has a `modules` object, keyed by the module's path relative to the repository root.
  Each module has the integers `num_branches` and `covered_branches`, and `missing_branches`, a list with one entry per uncovered branch.

## Fixtures: `tests/parity/fixtures/` and `tests/parity/test_fixtures.py`

- Every `*.json` file under `tests/parity/fixtures/` (searched recursively) is an object with a `cases` array.
- Each case has:
  - `id`, unique across all fixtures, matching `^[A-Za-z0-9_./-]+$`;
  - `source`, the repository-relative path of the test file it came from;
  - `operation`, a string;
  - `input`;
  - either `expected` or `expected_error`, an object with a string `code`.
- `test_fixtures.py` runs one pytest test per case, with the case's `id` as its pytest id, so the test's name ends in `[<id>]`.
- When `SDM_PARITY_FIXTURES_DIR` is set, it reads fixtures from that directory instead.

## PQL corpus: `tests/parity/pql/` and `tests/parity/test_pql_corpus.py`

- Every `*.json` file under `tests/parity/pql/` (searched recursively) is an object with a `cases` array.
- Each case has:
  - `id`, unique across the corpus, matching `^[A-Za-z0-9_./-]+$`;
  - `query`, the PQL text;
  - `schema`, the schema it's checked against, inline or as a reference;
  - for a valid query, `expected_tree`, its parse tree as JSON;
  - for an invalid query, `expected_error`, an object, or `expected_errors`, a non-empty array of objects.
    Each has a string `code` and a `span`: `null`, or an object with the integers `start_row`, `start_col`, `end_row` and `end_col`, as in `predictive_query.schema.common.Span`.
- `test_pql_corpus.py` runs one pytest test per case, with the case's `id` as its pytest id, so the test's name ends in `[<id>]`.
- When `SDM_PARITY_PQL_DIR` is set, it reads cases from that directory instead.
