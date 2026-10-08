# Phase 6 acceptance contract

The acceptance checks in this directory, and the Rust test `rust/crates/dialects/tests/accept_capture.rs`, test behavior through a small, fixed interface.
This file is that interface.
Everything it doesn't name is the implementer's choice.

## Names

- A **backend** is the name of a directory under `packages/sdm-relational-engine/src/sdm_relational_engine/rfm/backend/` that holds a `dialect.py`: `databricks`, `duckdb`, `postgres`, `snow` or `sqlite`.
  Lakebase scenarios use `postgres`, Snowflake scenarios use `snow`.
- A **function** is named as follows:
  - a function defined at the top level of the backend's `dialect.py`, public or private: its bare name, such as `rows_by_key` or `_keys`;
  - a function of `rfm/base/utils.py`: `utils.<name>`, such as `utils.ordered_sample_sql`;
  - `quote_ident` of `sdm_connectors/sql.py`: `sql.quote_ident`;
  - a function of `rfm/view_expr.py`: `view_expr.<name>`, such as `view_expr._unsafe_expr_reason`;
  - a **moved helper**, a method, property or cached property of a class defined in the backend's `table.py` or `sampler.py` that computes SQL or parameters: `<Class>.<attribute>`, such as `SnowTable._prefilter`.
- A **statement function** is one whose return annotation is `Statement`: it returns `(sql, params)`.

## Encoded values

Arguments and results are JSON, written by `_codec.py` beside this file, which the capture may import.
They extend the fixture tags of `tests/parity/FORMAT.md` (`$int`, `$float`, `$decimal`, `$timestamp`, `$datetime`, `$date`, `$timedelta_us`, `$bytes`, `$uuid`, `$na`, `$nat`, `$np`, `$ndarray`, `$tuple`) with:

| Tag | Means |
|---|---|
| `{"$time": ISO}` | A `datetime.time` |
| `{"$dtype": V}` | The `predictive_query` `Dtype` whose value is `V`, such as `"int"` |
| `{"$fraction": [N, D]}` | A `fractions.Fraction`; `N` and `D` are encoded integers |
| `{"$arrow": {"type": T, "values": [...]}}` | A `pyarrow.Array` of type `T`, spelled as `str(array.type)`; with `"chunks": [n, ...]`, a `pyarrow.ChunkedArray` of chunks of those lengths |
| `{"$arrow_type": T}` | A `pyarrow.DataType`, spelled as `str(type)` |
| `{"$series": {"dtype": D, "values": [...], "name": NAME}}` | A pandas `Series` with the default index; `D` is `str(series.dtype)` |
| `{"$record": {"class": C, "fields": {...}}}` | An instance of a `NamedTuple` or dataclass; `C` is `module.QualifiedName`, such as `sdm_relational_engine.rfm.base.source.KeyCountPart` |
| `{"$quote": C}` | A function of one name that quotes it as `quote_ident(name, C)` does, such as the `quote` that `utils.sample_order_terms` takes |
| `{"$object": {...}}` | A mapping with one key that starts with `$` |

The values of an `$arrow` array are, by type:

- integer, floating-point, boolean, `string` and `large_string`: each value as `to_pylist()` gives it, encoded;
- `timestamp`, `date32`, `date64`, `time32`, `time64` and `duration`: each value's integer storage, the count of the type's units since the epoch (UTC for a timestamp with a zone) or since midnight;
- `decimal128` and `decimal256`: each value as `str(Decimal)`;
- `binary` and `large_binary`: `{"$bytes": BASE64}`;
- nulls are `null`.

Other Arrow types are not part of the format.

A statement's `params` are encoded as a transcript step's are (`tests/parity/FORMAT.md`, "Encoded values", written by `scripts/parity/_parity/values.py`'s `encode`), so a captured statement and a recorded step compare directly.

## The capture

`python scripts/parity/capture_dialects.py --out DIR`, run from the repository root in the phase venv, writes `DIR/<backend>.json` for each of the five backends and nothing else; without `--out` it writes `tests/parity/dialects/`.
It needs no network, database or credentials, and writes the same bytes on every run, whatever `PYTHONHASHSEED` is.
The committed `tests/parity/dialects/*.json` are what it writes.

Each file is a JSON object:

| Key | Holds |
|---|---|
| `format` | The integer `1` |
| `backend` | The backend |
| `calls` | The captured calls, each an object as below |

| Key | Holds |
|---|---|
| `function` | The function, named as above |
| `args` | A JSON object of encoded arguments |
| `sources` | A non-empty list of where the call was made: `scenario:<name>` for the replay of the transcript whose `name` is `<name>`, `synthetic:<label>` for a call the capture makes itself |
| `sql`, `params` | For a statement function: the SQL text, and the encoded parameters, or `null` |
| `result` | For any other function: what it returned, encoded |

- For a function that is not a moved helper, `args` holds every parameter of its signature by name, defaults applied, as `inspect.signature(f).bind(...)` then `apply_defaults()` give them, so `f(**decoded_args)` makes the same call.
- For a moved helper, `args` holds what the capture chose to describe its inputs, and `result` is what the Python attribute gave.
- Only calls that returned are captured.
- Calls with the same `function` and `args` may share one entry, which then lists every source.
- A call made inside another captured call, such as `_keys` inside `rows_by_key`, is captured too.

### Moved helpers

`scripts/parity/capture_dialects.py` defines `replay_helper(backend, function, args)`.
Given a moved helper's `backend`, `function` and encoded `args` from the capture, it calls that very attribute of the engine's class, on an object it builds from `args` alone, and returns what the attribute returned.
The checks wrap the attribute on its class while `replay_helper` runs, and require that it was called and returned the captured `result`.

## The Rust crate

The package is `sdm-dialects` in `rust/crates/dialects`, with the library `sdm_dialects`.
Its root exports `render_json(backend: &str, function: &str, args_json: &str) -> Result<String, E>`, where `E` implements `Display`:

- `args_json` is a call's `args`, as JSON text;
- `Ok` holds JSON text: `{"sql": ..., "params": ...}` for a statement function, `{"result": ...}` for any other, encoded as in the capture;
- `Err` is for an unknown backend or function, or arguments it can't read.

`accept_capture.rs` reads every `tests/parity/dialects/*.json` and requires each call's output equal to the captured one: the SQL byte for byte, and `params` or `result` equal by value and JSON type.

## The Python binding

`sdm_core._native.dialect_render(backend, function, args_json)` takes `str` arguments as `render_json` does and returns a 2-tuple:

- for a statement function, `(sql, params_json)`, where `params_json` is the encoded parameters as JSON text, `null` when there are none;
- for any other function, `(None, result_json)`.

It raises `ValueError` for an unknown backend or function and for arguments it can't read, and never ends in a Rust panic.

Functions not tied to one backend (`utils.*`, `sql.quote_ident`, `view_expr.*`) answer the same under every backend.
Besides the captured functions, it answers `utils.random_state.randint`, whose `args` are `random_seed`, `low`, `high` and `size`, with `result` the list `random_state(random_seed).randint(low, high, size=size).tolist()` gives.

The checks build the binding's wheel from this checkout with maturin and install it into the phase venv before calling it (`_rc.install_local_wheel`).

## What the capture must hold

Every check below is in `accept_capture.py`; the names in it are the ones the brief and `investigations/RESEARCH-p6.md` give.

- **Every function.**
  Each backend's file has a call to every top-level function of its `dialect.py`; to every `utils.*` function among `ordered_sample_sql`, `key_count_select`, `count_distinct_values`, `repeated_key_select`, `sample_order_terms`, `percent_literal` and `prefilter_fraction` that its `dialect.py` or `table.py` imports; to `sql.quote_ident` if its `dialect.py` imports it; and to each of its moved helpers in this table:

  | Backend | Moved helpers |
  |---|---|
  | `databricks` | `DatabricksSampler._bound_format`, `DatabricksTable._is_large` |
  | `duckdb` | `DuckDBTable._is_zoned_time` |
  | `postgres` | `PostgresSampler._utc_bounds`, `PostgresTable._prefiltered_sample_source` |
  | `snow` | `SnowSampler._zone`, `SnowSampler._bound_format`, `SnowTable._prefilter`, `SnowTable._listing` |
  | `sqlite` | `SQLiteSampler._probe_time_spelling`, `_TimeSpelling.spell` |

  Others may be added under the same naming.
- **Every scenario.**
  For every transcript under `tests/parity/transcripts/` over a database (not `memory` or `tabular`), the live ones included, every `execute` step's SQL and encoded parameters equal those of a captured statement, of the matching backend, whose `sources` names that scenario.
- **The synthetic branches.**
  - `holds_any`, in every backend, with two values or more, and with none.
  - Composite keys, in every backend: `key_counts` with a part of two columns or more, and `repeated_key` with two columns or more.
  - Expression columns, in every backend: the inference sample (`first_rows` on DuckDB and SQLite, `sample` elsewhere) with a projection `... AS ...`.
  - Empty inputs: `rows_by_key` with no ids in every backend, and `chunk_rows` with no rows on PostgreSQL and Databricks.
  - Key lists of more than one 900 KB chunk: `chunk_rows` at `max_bytes` 900,000 giving two chunks or more, on PostgreSQL and Databricks.
  - The three scan caps and their fallbacks:
    PostgreSQL `PostgresTable._prefiltered_sample_source` giving a block sample and giving `None`, and `sample` over a `TABLESAMPLE SYSTEM` source;
    Snowflake `SnowTable._prefilter` giving a clause and giving `None`, `SnowTable._listing` giving a listing and giving `None`, `sample` over a `SAMPLE SYSTEM (` source, and a seeded `entity_rows` with a `prefilter`;
    Databricks `DatabricksTable._is_large` giving `true` and `false`, and `sample` over a `bounded` source;
    and `utils.prefilter_fraction` giving a fraction and giving `None`, on PostgreSQL and on Snowflake.
  - Zoned bounds:
    PostgreSQL `serialize_time` with `utc` giving `+00:00`, `window` with `time_type` `timestamp with time zone`, and `PostgresSampler._utc_bounds` giving `true` and `false`;
    Databricks `DatabricksSampler._bound_format` giving both formats, and `window` with a bound ending `+00:00`;
    Snowflake `zone_of` giving `LTZ`, `TZ` and `null`, `window` and `neighbors` each with `zone` `LTZ` and with `TZ`, `SnowSampler._zone` giving `LTZ`, `TZ` and `null`, and `SnowSampler._bound_format` giving both formats;
    DuckDB `time_column` with `zoned`, `_duckdb_type` giving `TIMESTAMPTZ`, `window` with zoned end times, and `DuckDBTable._is_zoned_time` giving `true` and `false`.
  - String time columns: `serialize_time` without `utc` on PostgreSQL, and `SQLiteSampler._probe_time_spelling` giving two different spellings.
