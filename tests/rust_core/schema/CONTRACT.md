# Phase 6.5 acceptance contract

The acceptance checks in this directory, and the Rust test `rust/crates/schema/tests/accept_fixtures.rs`, test behavior through a small, fixed interface.
This file is that interface.
Everything it doesn't name is the implementer's choice.

The checks build the binding's wheel from this checkout with maturin and install it into the phase venv before calling it (`_rc.install_local_wheel`).
No call into the binding may end in a Rust panic: a panic fails the check that made it.

## Outcomes

A schema outcome is what one `Schema.from_data` or `Schema.from_source` call gives, in one of two forms:

- `{"schema": D, "warnings": W}`, when it returns a schema;
- `{"error": {"code": C, "message": M}, "warnings": W}`, when it refuses.
  `C` is the error's `code` where it has one (every `StructuredDataError` does), otherwise its class name, and `M` is its `message` where it has one, otherwise its text.

`W` is the list of `UserWarning`s, subclasses included, that the call gave, in order, repeats included, each `{"category": "module.QualifiedClass", "message": text}`, as in a transcript's `warnings` (`tests/parity/FORMAT.md`).

`D`, the description, is the `_describe_schema` shape of `tests/parity/_fixture_ops.py`, with more of each table:

```
{
  "tables": {
    NAME: {
      "primary_key": COLUMN or [COLUMN, ...] or null,
      "time_column": COLUMN or null,
      "duplicate_keys": the table's duplicate_keys (an integer, or null),
      "source_name": the SourceTable's source_name (only for from_source),
      "columns": {
        COLUMN: {"dtype": str(dtype), "stype": str(stype), "untyped": bool, "stype_declared": bool}
      }
    }
  },
  "relationships": [[source_table, source_column, target_table, target_column], ...],
  "definition": {
    "tables": {
      NAME: {
        "cols": [[name, stype, dtype, timestamp_format], ...],
        "pkey": ..., "time_col": ..., "end_time_col": ...
      }
    },
    "col_groups": [[[table, column], [table, column]], ...],
    "warnings": W
  }
}
```

`definition` is `to_schema_definition(schema)`, with enums by their values, and the warnings it gave, in `W`'s form.
Tables and columns are compared in order as well as by value; everything else is compared by value and JSON type.

`_schema.py`'s `python_outcome` and `describe_schema` are the reference for this form.

## The Python describer

`sdm_core.describe`, a Python module in the `sdm-core` wheel (`rust/bindings/python/python/sdm_core/describe.py`, or a package of that name), is the host's describer (DV19):

- `describe_frames(frames)` takes a mapping of table names to pandas `DataFrame`s and returns `(tables, column_tags_json)`:
  - `tables` is a `dict` of the same names, in the same order, each a `pyarrow.Table`;
  - `column_tags_json` is JSON text: an object of table names, each an object of column names to that column's tag, any JSON value.
    Every column whose pandas dtype is `object` has a tag that is not `null`.
- The same frames give the same tags, whatever `PYTHONHASHSEED` is.
- It works with `sdm_client`, `sdm_relational_engine` and `predictive_query` unimportable: it describes a column's contents, and never asks today's inference what the column is.

## The Python binding

`sdm_core._native` holds, besides what earlier phases put there:

### `schema_from_data(tables, column_tags_json, options_json)`

`Schema.from_data`, in the core.
`tables` and `column_tags_json` are what `describe_frames` gave.
`options_json` is a JSON object:

| Key | Holds |
|---|---|
| `primary_keys` | `null`, or `{table: column or [column, ...]}` |
| `time_columns` | `null`, or `{table: column}` |
| `relationships` | `null` to infer them, or a list of `[source_table, source_column, target_table, target_column]` |
| `edits` | A list, possibly empty, of `{"table", "column", "set", "value"}`, applied in order after inference as `setattr(schema[table][column], set, value)` would be |

It returns `(schema_json, warnings_json)`, both JSON text: `schema_json` is `D`, or `{"error": {"code", "message"}}` for a refusal; `warnings_json` is `W`.
An edit that is refused makes the outcome that refusal, with the warnings given until then.

### `schema_from_source(backend, source_json, args_json)`

`Schema.from_source`, as a run of the core (Phase 4's run loop).

- `backend` is the dialect, as Phase 6 names it: `databricks`, `duckdb`, `postgres`, `snow` or `sqlite`.
- `source_json` is `{"namespace": source.namespace, "semantic_model": NAME or null, "trust_expressions": bool}`; for a semantic model, its `name` and `trust_expressions`, and its source's namespace.
- `args_json` is `{"tables", "primary_keys", "time_columns", "relationships", "check_keys"}`: the call's keyword arguments, as for `schema_from_data`, `tables` being `null` or `{name: name in the source}` and `check_keys` a boolean.
- It returns a `Run`, driven with `next()` and `submit(id, outcome)`.
  The host runs each query as `source._run(query.sql, query.params or None)` on the Python source the call was given, and submits `Outcome.data(table)` with what it returned, or, when it raised, `Outcome.error(sqlstate, vendor_code, message)` with the error's `sqlstate` (when a `str`), its `errno` (when an `int`) and its text.
  A run may ask several queries in a step, in any order.
- The value of its `done` step is the 2-tuple `schema_from_data` returns: `(schema_json, warnings_json)`.

### `dates_op(op, values_json, options_json)`

The `dates.*` and `time_column.read` operations of `tests/parity/fixtures/dates.json`, and `infer.timestamp`:

- `op` is the operation's name; `values_json` is the input's `values` as JSON text, `null` when it has none; `options_json` is a JSON object of the input's other keys.
- `infer.timestamp`, with options `{"column": NAME}`, is `contains_timestamp(pd.Series(values, dtype=object), NAME, Dtype.string)` of `sdm_relational_engine.rfm.metadata_inference`.
- It returns JSON text: `{"output": X}`, where `X` is what the fixture runner (`tests/parity/_fixture_ops.py`) gives, or `{"error": {"code", "message"}}` where it raises.
  Date parsing is pinned to the phase venv's pandas, 3.0.6 (DV20).

### `composite_op(op, input_json)`

The `composite.*` operations of `tests/parity/fixtures/composite-key.json`: `input_json` is the case's whole `input`, and it returns JSON text as `dates_op` does.

### `metric_view_parse(yaml, trust)`

`parse_metric_view(yaml, trust_expressions=trust)` of `sdm_relational_engine.rfm.backend.databricks.metric_view`, as JSON text:

- `{"spec": S}`, where `S` is `dataclasses.asdict` of the `MetricViewSpec` with every tuple as an array;
- `{"error": {"code", "message"}}` where it raises: the error's class name, such as `ValueError` or `UnsupportedFilterError`, and its text.

### `semantic_view_parse(yaml, name)`

The brief's parser of a Snowflake semantic view's YAML.
Its output is the implementer's design; it returns a `str` or raises an `Exception`, never panics.
Semantic views are compared through `schema_from_source` instead (below).

### Refusals

Each of these functions raises `ValueError`, never panics, for text that isn't the JSON it takes, an unknown `backend` or `op`, or arguments it can't read.

## The Rust crate

The package is `sdm-schema` in `rust/crates/schema`, with the library `sdm_schema`.
Its root exports `fixture_op(operation: &str, input_json: &str) -> Result<String, E>`, where `E` implements `Display`:

- it answers the operations `dates_op` and `composite_op` answer, `input_json` being a fixture case's whole `input`;
- `Ok` holds the JSON text `dates_op` and `composite_op` return;
- `Err` is for an unknown operation, or an input it can't read.

`accept_fixtures.rs` reads every case of `tests/parity/fixtures/dates.json` and `composite-key.json` and requires `fixture_op` to give the expected output by value and JSON type, or the expected error code, and message where the fixture has one.

## What the checks compare

- **Schemas** (`accept_schemas.py`).
  Every transcript under `tests/parity/transcripts/` is replayed through today's Python, keeping every `Schema.from_data` and `Schema.from_source` call its public calls make (a `Request` or a `PredictiveQuery` over frames makes one too), with its outcome and the recorded steps it took.
  The core's outcome for the same call must be equal.
  A `from_source` run is answered from those recorded steps: each query by the `execute` step with the same SQL, byte for byte, and parameters of the same encoding, each step at most once, in any order; and the run must ask for every one of them.
- **Fixtures** (`accept_fixtures.py`, `accept_fixtures.rs`).
  Every case of `time-and-keys.json`, `relationships.json`, `dates.json` and `composite-key.json`, and every `wire.infer_dtype` case of `wire.json`, gives its expected output.
  `schema.from_data` cases compare `D` reduced to the fixture's shape, and the warnings' messages; the whole outcome, warning categories included, must equal today's Python's too.
  A `wire.infer_dtype` case is a one-column table `t` with the column `c`: its inferred wire dtype is `""` when the column is `untyped`, otherwise its `dtype`.
- **Date columns** (`accept_dates_series.py`): 100,000 generated text columns (seed 20261007) give through `dates_op` exactly what today's Python gives, for every `dates.*` operation that reads a column, `time_column.read`, `infer.timestamp`, and `dates.read` with the conversion `dates.conversion_that_works` found, where it found one (its `arguments` as `{"utc", "dayfirst", "mixed"}`).
- **Views** (`accept_views.py`).
  Metric views: every definition the engine's tests parse, every recorded one, and 10,000 generated ones (seed 20261008, with and without trust) give through `metric_view_parse` what `parse_metric_view` gives.
  Semantic views: every definition the engine's tests read, every recorded one, and 10,000 generated ones (seed 20261009) give, through `schema_from_source` over the stand-in warehouse of `_warehouse.py`, the outcome today's `Schema.from_source` gives over it, from the same statements.
  The stand-in answers like Snowflake: a quoted name is the name it spells, case kept; an unquoted one is upper-cased.
- **The quoting bug** (`accept_semantic_quoting.py`).
  The recorded semantic view's YAML, with every name of the shop written quoted and in lower case (`name: '"orders"'`), over the shop written so, is read by `Schema.from_source` as the recorded view is read over the stand-in, with every table and column name in lower case: the tables `orders`, `products` and `users`, the shop's columns, keys and links, and no warning.
  Its SQL quotes each name once (`DESCRIBE TABLE "TEST"."PARITY_SHOP"."orders"`).
  The core gives the same outcome over the same statements.
- **The describer** (`accept_describer.py`): every `object` column of every fixture under `tests/parity/fixtures/`, and the shapes listed in that file, alone and beside a key, give through the describer and `schema_from_data` the outcome today's `Schema.from_data` gives.
- **Hash seeds** (`accept_hashseed.py`): run under `PYTHONHASHSEED` 1 and 4242, the core's outcomes for the corpus, the fixtures, the describer's shapes and the first generated views and date columns, and today's Python's for the corpus and the fixtures, are identical.
- **The engine's pytest configuration** (`accept_pytest_config.py`, task 5): copied beside a probe test, it lets pyOpenSSL's `X509.get_subject is deprecated` `DeprecationWarning` pass, and still fails on any other `DeprecationWarning`.
- **The recorder's Snowflake sign-in** (`accept_recorder_signin.py`, task 6): with `snowflake.connector.connect` faked, `live.prepare` and `live.connection` sign in with `authenticator='PROGRAMMATIC_ACCESS_TOKEN'` and the token when `SNOWFLAKE_TOKEN` is set, with the password otherwise, and `live.prepare` refuses with `CannotRecord`, naming `SNOWFLAKE_TOKEN or SNOWFLAKE_PASSWORD`, when neither is set.
- **On its own** (`accept_independent.py`): the core gives the same answers in a process where `sdm_client`, `sdm_relational_engine` and `predictive_query` can't be imported (schemas from frames and from semantic views), and where pandas, numpy, dateutil, PyYAML and pytz can't be imported either (`dates_op`, `composite_op`, `metric_view_parse`): it answers by itself, not by calling today's Python.
  So `sdm_core._native` imports none of these to answer, and `sdm_core.describe` imports none of the first three.
