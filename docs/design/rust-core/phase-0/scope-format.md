# Scope manifest and case inventory format

Task T0.5 of [Phase 0](design.md).
Each language has one scope manifest: [scope-python.yaml](scope-python.yaml) and [scope-java.yaml](scope-java.yaml).
A manifest lists the public paths and the type x role cells in scope for that language, and the contract case IDs that cover each one.
It is the input to the L2 coverage matrix (plan section 6): for the current slice, every path and every cell needs at least one case, and an empty one fails.
The JSON Schema is [scope.schema.json](scope.schema.json).

## Shape

A manifest is one YAML document with seven keys, all required.

| Key | Holds |
|---|---|
| `format` | The literal `sdm-scope/1`; a change to this format bumps it |
| `language` | `python` or `java` |
| `slices` | `S0` to `S3`, each with a one-line description from plan section 4 |
| `backends` | The backends this language supports, each an `id` and a `what` |
| `operations` | The public operations, each an `id`, its `api` in this language, its `layer` and its `modes` |
| `paths` | One entry per operation x mode x backend in scope, with its `slice` and `cases` |
| `cells` | One entry per Arrow type x column role, with its `expect` and `cases` |

Any entry may carry a `note`.

### Backends

A backend is where a path's data comes from, named by the provider the host runs, not by the vendor alone.
`databricks-sql` and `databricks-spark` are separate backends because they reach the core through different drivers.
A semantic model (`databricks-metric-view`, `snowflake-semantic-view`) is a backend of its own, because it is a source of its own in the public API.
`frames` is pandas DataFrames in memory, whatever they were read from.
`local-file` and `s3` are only the backends of `connectors.read`; the frames it returns are `frames`.
`any` marks behavior that does not depend on the backend: a case on any backend covers it.
The allowed IDs are the union across languages, fixed in the schema.

### Operations and modes

An operation ID is language-neutral (`query.materialize`, `client.predict`), so the two manifests can be compared row by row.
`api` names the public symbols in this language, or is `null` until that language's API exists.
`layer` is `core` when the Rust core implements the behavior, and `host` when the language binding keeps it (presentation, HTTP, opening connections, reading files).
Host paths are in the manifest because they must not regress either, and because the Databricks jobs exercise them; the routing boundary only switches `core` paths.
A mode is the input shape that changes which code runs: for `query.materialize`, the kind of question and whether rows are named by `indices` or passed as `predict`.
Options that only change values inside one route (`anchor_time`, `num_context_samples`, `inference_config`, `outputs`, the model handle) are case variants, not modes.

### Paths

A path ID is `<operation>/<mode>/<backend>`, the plan's "one public operation with one input mode on one backend".
`slice` is the slice whose gate the path must pass before it switches; S0 and S1 and S2 paths switch together after S2, and S3 paths after S3 (plan section 4).
`cases` lists the contract case IDs covering the path.

### Cells

A cell ID is `<type>/<role>`.
Types are the closed Arrow type set from plan section 3, with timestamps split by unit and zone: `int8` to `uint64`, `float32`, `float64`, `decimal128`, `decimal256`, `utf8`, `binary`, `date32`, `timestamp-ns`, `timestamp-us`, `timestamp-ns-tz`, `timestamp-us-tz`, `duration-ns`, `duration-us`, `time64`, `list`, `struct`, `map`, `dictionary-utf8`, `bool`, `null`.
Roles are the parts a column plays in a request: `primary-key`, `foreign-key`, `time-column`, `feature`, `target`.
`expect` is `accept`, `refuse`, or `unknown`.
`accept` and `refuse` say what today's Python client does; `unknown` means nobody has checked yet.
Host-side conversions (`large_utf8`, `timestamp[s]`, `large_list` and the like) are checked by L5 against the conversion tables, not here: a cell names the type the core receives.
Every cell applies from S1, since S0 reads no data.

### Case IDs

Case IDs are the IDs of the shared L2 contract corpus, so one ID means the same case in every language.
They match `^[a-z0-9][a-z0-9._-]*$`.
Every `cases` list is empty until Phase 1a creates the corpus.
A case may cover several paths and several cells, and is listed under each.

## Checks

The schema checks shape: keys, enums, the ID patterns, and unique case IDs within a list.
These referential rules are checked by the coverage generator Phase 1a writes (they were checked by a one-off script for the first manifests):

1. Backend, operation, path and cell IDs are unique within a manifest.
2. Each path's operation is declared, its mode is one of that operation's modes, and its backend is declared.
3. Every declared operation, mode and backend appears in at least one path.
4. From Phase 1a on, every case ID resolves to a case in the L2 corpus.
5. At a slice gate, every path of that slice or an earlier one has at least one case, and so does every cell whose `expect` is not `unknown`.
6. No cell is `unknown` at the S1 gate: Phase 1a and Phase 1b record the reference's outcome for each one, and a cell the core refuses where the reference accepted needs an owner-approved register entry.
7. A path or cell is removed only with an owner decision, since dropping one is the kind of regression decision 3 names.

Validate the shape with:

```text
uvx --from jsonschema --with pyyaml python -c "import json,sys,yaml,jsonschema; s=json.load(open('scope.schema.json')); [jsonschema.validate(yaml.safe_load(open(f)), s) for f in sys.argv[1:]]" scope-python.yaml scope-java.yaml
```

## What the first manifests contain

Python has 281 paths (S0 2, S1 58, S2 61, S3 160) and Java 66 (S0 2, S1 46, S2 18); each has 140 cells.
The operations come from `sdm_client.__all__`, `sdm_connectors.__all__` and `sdm_connectors.databricks.__all__`, as recorded in `tests/golden/public_api.json`.
Python covers every backend of plan decision 2: DuckDB, SQLite, Postgres, Lakebase, Databricks over a warehouse and over Spark with metric views, Snowflake with semantic views, in-memory frames, and files read from Amazon S3.
Java covers DuckDB, SQLite, Postgres and Databricks over a SQL warehouse, with no frames, sessions, tabular requests or S3 paths, since those wait until the owner adds Java scope.

Choices worth a second look:

- `query.validate` has no public method behind it: today PQL is validated inside `materialize()`, so S0 observes validation as the refusal raised before data is read, or as the resolved query.
- Classification is split into `binary` and `multiclass`, because the jobs ask both (`inactivity_90d` and `product_color`) and their results have different columns.
- `predict` and `fit` paths are per backend, following the plan's example path ("`predict` of a regression query on Snowflake").
- Ranking and forecasting on Databricks and Lakebase are S3, as the plan says, but the S2 job matrix needs them; those paths carry a note pointing at [databricks-jobs.md](databricks-jobs.md), and the owner decides (J1 there).
- Java's metric-view paths and its `databricks-serving` connection are assumptions, noted on each path, until the owner and the Java consumer confirm them.
- A custom (non-SQL) data provider is not a path in either manifest: the S1 test provider and the swapped data layer are Rust-level checks, and a public provider API in Java is a question for when Java's API is designed.
- Most cells outside the obvious ones are `unknown`; Phase 1b's case inventory (T0.7) and Phase 1a's runners resolve them against the reference.
