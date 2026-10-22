# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Records T3.4's Databricks fixtures from the reference, before C-1 (T3.4's
test author; the T3.4 brief's exits 2, 4 and 5).

Run with the reference checkout's interpreter, offline:

    ~/code/sdmc-reference/.venv/bin/python -I \
        contract/fixtures/databricks/record.py [--check]

No warehouse is reached. The old client (``sdm_client``, ``sdm_connectors``
and the relational engine at the reference commit) runs each case below on a
``DatabricksSource`` over a stand-in connection: a test double of a
``databricks.sql`` connection that records every statement the old client
sends, with its parameters, and answers it from a local DuckDB holding the
case's base and setup, the tables typed as a Databricks catalog types them
(:data:`DATABRICKS_TYPES`, this fixture's stand-in for T3.3's type map). The
catalog's ``information_schema`` statements are answered from DuckDB's
catalog in Unity Catalog's shape; every other statement is rewritten into
DuckDB's SQL (:func:`duckdb_text`) and run there, its ``TABLESAMPLE`` clause
dropped (only the warehouse draws; C-1 records the draw).

It writes, next to itself:

- ``capture.json``: the fixture capture, shaped as the design's
  ``sdm-capture/1`` (choice 2): per case and leg, every statement the old
  client sent with its parameters and its result, each result as rows and
  as Arrow IPC (base64), the shape a driver's ``fetchall_arrow`` gives;
  and, for each catalog answer, how the old client read it
  (``DatabricksTable._to_dtype`` and its key and foreign-key rules), the
  expectation of the core's catalog reader. C-1's capture replaces it;
  ``"fixture": true`` marks it as not recorded on a warehouse.
- ``reference-statements.json``: for each request kind of
  ``sdm_contract::data::DataRequest``, the statements the old client sent
  for it in these runs, the reference side of T3.4's goldens.

``--check`` writes nothing and fails if either file differs. ``--diff``
writes nothing and prints each golden of
``rust/crates/sdm-data/tests/databricks_goldens.rs`` beside the reference's
statement of its kind (the same table where the reference has one), as a
unified diff: T3.4's evidence of how the core's statements differ from the
old client's (the brief's exit 2).
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import decimal
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASES = ROOT / 'contract' / 'bases.yaml'
CASES = ROOT / 'contract' / 'cases'
REFERENCE_COMMIT = '43e55f4f'
CATALOG = 'main'
SCHEMA = 'shop'
LEG = 'databricks-sql'

#: The cases recorded, each with the Schema.from_source arguments and the
#: materialize call of its case file (``until`` as the case's).
RECORDED = (
    's1/base.binary.json',
    's1/key.declared-int32.json',
    's2/databricks.key.decimal-38-6.json',
    's2/databricks.name.odd-columns.json',
    's2/databricks.name.caller-case.json',
    's2/databricks.value.struct-feature.json',
    # Recorded for their statements only (reference-statements.json), each
    # a run of the shared cases' kinds the ones above do not send.
    's2/warehouse.shop.ranking.json',
    's2/warehouse.shop.forecast.json',
    # Two runs no case file holds yet, for the two request kinds the cases
    # above never reach: a key the catalog does not keep unique repeating
    # once (repeated_key, after key_counts), and a table past the preview's
    # 1,000 rows, whose time column is chosen over every row (column_ranges,
    # `_full_range`).
    {
        'id': 'fixture.key-repeating-once',
        'until': 'from_source',
        'input': {
            'base': 'shop',
            'setup': ['INSERT INTO users SELECT * FROM users LIMIT 1'],
        },
    },
    {
        # Every column type the catalog reader reads a data type from, as
        # the old client's `DatabricksTable._to_dtype` reads each; the map
        # column makes the old client refuse the table, after its catalog
        # statements.
        'id': 'fixture.every-type',
        'until': 'from_source',
        'input': {
            'base': 'shop',
            'setup': [
                'CREATE TABLE kinds AS SELECT 1::TINYINT AS a_tinyint, '
                '2::SMALLINT AS a_smallint, 3::INTEGER AS an_int, '
                '4::BIGINT AS a_bigint, 1.5::FLOAT AS a_float, '
                '2.5::DOUBLE AS a_double, true AS a_boolean, '
                "'x' AS a_string, 'ab'::BLOB AS a_binary, "
                "DATE '2024-01-02' AS a_date, "
                "TIMESTAMP '2024-01-02 03:04:05' AS a_timestamp_ntz, "
                "TIMESTAMPTZ '2024-01-02 03:04:05+00' AS a_timestamp, "
                '12345678901234567890123456789012345678::DECIMAL(38, 0) AS a_decimal_38_0, '
                '1.25::DECIMAL(10, 2) AS a_decimal_10_2, '
                '[1, 2]::BIGINT[] AS an_array_of_bigint, '
                '[1.5]::DOUBLE[] AS an_array_of_double, '
                "['a']::VARCHAR[] AS an_array_of_string, "
                "MAP {'k': 1} AS a_map"
            ],
        },
    },
    {
        'id': 'fixture.past-the-preview',
        'until': 'from_source',
        'input': {
            'base': 'shop',
            'setup': [
                'CREATE TABLE visits AS SELECT i::BIGINT AS visit_id, '
                "(i % 8)::BIGINT AS user_id, TIMESTAMP '2023-01-01' + "
                'to_days((i % 400)::INTEGER) AS ts FROM range(1200) t(i)'
            ],
        },
    },
)

#: The runs recorded only for reference-statements.json, kept out of the
#: capture to keep it small.
STATEMENTS_ONLY = (
    's2/warehouse.shop.ranking.json',
    's2/warehouse.shop.forecast.json',
)

#: DuckDB's declared types, as a Databricks catalog types the column
#: (``full_data_type``): this fixture's stand-in for T3.3's type map.
DATABRICKS_TYPES = {
    'BIGINT': 'bigint',
    'INTEGER': 'int',
    'SMALLINT': 'smallint',
    'TINYINT': 'tinyint',
    'DOUBLE': 'double',
    'FLOAT': 'float',
    'BOOLEAN': 'boolean',
    'VARCHAR': 'string',
    'DATE': 'date',
    'TIMESTAMP': 'timestamp_ntz',
    'TIMESTAMP WITH TIME ZONE': 'timestamp',
    'BLOB': 'binary',
}


def databricks_type(duckdb_type: str) -> str:
    r"""``duckdb_type`` as Unity Catalog's ``full_data_type`` writes it."""
    text = duckdb_type.strip()
    if text in DATABRICKS_TYPES:
        return DATABRICKS_TYPES[text]
    if match := re.fullmatch(r'DECIMAL\((\d+),\s*(\d+)\)', text):
        return f'decimal({match[1]},{match[2]})'
    if text.endswith('[]'):
        return f'array<{databricks_type(text[:-2])}>'
    if match := re.fullmatch(r'STRUCT\((.*)\)', text):
        fields = []
        for part in _split_top(match[1]):
            name, _, kind = part.strip().partition(' ')
            fields.append(f'{name.strip(chr(34))}:{databricks_type(kind)}')
        return f'struct<{",".join(fields)}>'
    if match := re.fullmatch(r'MAP\((.*)\)', text):
        key, value = _split_top(match[1])
        return f'map<{databricks_type(key)},{databricks_type(value)}>'
    raise ValueError(f'no Databricks type for {duckdb_type}')


def _split_top(text: str) -> list[str]:
    parts, depth, start = [], 0, 0
    for at, char in enumerate(text):
        depth += {'(': 1, ')': -1}.get(char, 0)
        if char == ',' and depth == 0:
            parts.append(text[start:at])
            start = at + 1
    parts.append(text[start:])
    return parts


def _arrow_type(pa: Any, full_data_type: str) -> Any:
    r"""The Arrow type ``databricks-sql-connector`` fetches a column of
    ``full_data_type`` as."""
    kinds = {
        'bigint': pa.int64(),
        'int': pa.int32(),
        'smallint': pa.int16(),
        'tinyint': pa.int8(),
        'double': pa.float64(),
        'float': pa.float32(),
        'boolean': pa.bool_(),
        'string': pa.string(),
        'date': pa.date32(),
        'timestamp_ntz': pa.timestamp('us'),
        'timestamp': pa.timestamp('us', tz='Etc/UTC'),
        'binary': pa.binary(),
    }
    if full_data_type in kinds:
        return kinds[full_data_type]
    if match := re.fullmatch(r'decimal\((\d+),(\d+)\)', full_data_type):
        return pa.decimal128(int(match[1]), int(match[2]))
    return None


# --- the stand-in connection ----------------------------------------------------


def duckdb_text(text: str) -> str:
    r"""A statement of the old client's Databricks SQL as DuckDB runs it: the
    catalog and schema dropped from a name, each backtick-quoted name double
    quoted, ``posexplode(from_json(?, '<type>'))`` as DuckDB's ``unnest`` of
    the bound JSON, and the ``TABLESAMPLE`` clause dropped."""
    text = text.replace(f'`{CATALOG}`.`{SCHEMA}`.', '')
    text = re.sub(r'\nTABLESAMPLE \([^)]*\)( REPEATABLE \(-?\d+\))?', '', text)

    def explode(match: re.Match[str]) -> str:
        kind = match[1]
        element = re.fullmatch(r'array<(.*)>', kind)[1]
        if element.startswith('struct<'):
            fields = re.findall(r'(\w+):(\w+)', element[len('struct<') : -1])
            structure = {
                name: 'TIMESTAMP'
                if t == 'string' and name in ('e', 's')
                else _duck_scalar(t)
                for name, t in fields
            }
        else:
            structure = _duck_scalar(element)
        shape = json.dumps([structure])
        return (
            '(SELECT generate_subscripts(a, 1) - 1 AS pos, unnest(a) AS col '
            f"FROM (SELECT from_json(?::JSON, '{shape}') AS a))"
        )

    text = re.sub(r"posexplode\(from_json\(\?, '([^']*)'\)\)", explode, text)
    out, at = [], 0
    while at < len(text):
        char = text[at]
        if char == '`':
            end = at + 1
            name = []
            while True:
                if text[end] == '`' and text[end + 1 : end + 2] == '`':
                    name.append('`')
                    end += 2
                elif text[end] == '`':
                    break
                else:
                    name.append(text[end])
                    end += 1
            out.append('"' + ''.join(name).replace('"', '""') + '"')
            at = end + 1
        elif char == "'":
            end = text.index("'", at + 1)
            while text[end + 1 : end + 2] == "'":
                end = text.index("'", end + 2)
            out.append(text[at : end + 1])
            at = end + 1
        else:
            out.append(char)
            at += 1
    return ''.join(out)


def _duck_scalar(kind: str) -> str:
    return {'bigint': 'BIGINT', 'double': 'DOUBLE', 'string': 'VARCHAR'}[kind]


class Cursor:
    def __init__(self, connection: Connection) -> None:
        self._connection = connection
        self._table: Any = None

    def __enter__(self) -> Cursor:
        return self

    def __exit__(self, *exc: object) -> None:
        self._table = None

    def close(self) -> None:
        self._table = None

    def execute(self, operation: str, parameters: Any = None) -> Cursor:
        self._table = self._connection.answer(operation, list(parameters or []))
        return self

    def fetchall_arrow(self) -> Any:
        return self._table

    def fetchall(self) -> list[tuple[Any, ...]]:
        return _rows(self._table)

    def fetchone(self) -> tuple[Any, ...] | None:
        rows = self.fetchall()
        return rows[0] if rows else None


def _driver_connection() -> type:
    from databricks.sql.client import Connection as Driver

    return Driver


class Connection(_driver_connection()):  # type: ignore[misc]
    r"""A ``databricks.sql`` connection's test double over DuckDB: the
    driver's class, so the old client takes it as one, with none of its
    state."""

    def __init__(self, duck: Any) -> None:
        self._duck = duck
        self.log: list[dict[str, Any]] = []

    def __del__(self) -> None:
        pass

    def cursor(self) -> Cursor:
        return Cursor(self)

    def close(self) -> None:
        pass

    def answer(self, text: str, parameters: list[Any]) -> Any:
        import pyarrow as pa

        if text == 'SELECT current_catalog(), current_schema()':
            table = pa.table(
                {'current_catalog()': [CATALOG], 'current_schema()': [SCHEMA]}
            )
        elif (
            'information_schema.tables' in text
            and 'information_schema.columns' not in text
        ):
            table = self._tables()
        elif 'information_schema.columns' in text:
            table = self._columns(parameters[1])
        elif "constraint_type = 'FOREIGN KEY'" in text:
            table = self._foreign_keys(parameters[1])
        elif "constraint_type IN ('PRIMARY KEY', 'UNIQUE')" in text:
            table = self._keys(parameters[1])
        else:
            rewritten = duckdb_text(text)
            table = self._duck.execute(rewritten, parameters).to_arrow_table()
            table = self._as_driver(table)
        self.log.append(
            {'text': text, 'parameters': parameters, 'result': table}
        )
        return table

    # The catalog, in Unity Catalog's shape.

    def _declared(self, table: str) -> list[tuple[str, str, bool]]:
        rows = self._duck.execute(
            'SELECT column_name, data_type, is_nullable FROM '
            'information_schema.columns WHERE lower(table_name) = ? '
            'ORDER BY ordinal_position',
            [table.lower()],
        ).fetchall()
        return [
            (name, kind, nullable == 'YES') for name, kind, nullable in rows
        ]

    def _tables(self) -> Any:
        import pyarrow as pa

        names = [
            row[0]
            for row in self._duck.execute(
                'SELECT table_name FROM information_schema.tables '
                "WHERE table_type = 'BASE TABLE' ORDER BY table_name"
            ).fetchall()
        ]
        return pa.table({'table_name': pa.array(names, pa.string())})

    def _columns(self, table: str) -> Any:
        import pyarrow as pa

        rows = self._declared(table)
        keyed = {
            column
            for _, column in self._constraint_columns(table, 'PRIMARY KEY')
        }
        return pa.table(
            {
                'column_name': pa.array([r[0] for r in rows], pa.string()),
                'full_data_type': pa.array(
                    [databricks_type(r[1]) for r in rows], pa.string()
                ),
                # Unity Catalog's primary key columns are NOT NULL.
                'is_nullable': pa.array(
                    [
                        'NO' if (r[0] in keyed or not r[2]) else 'YES'
                        for r in rows
                    ],
                    pa.string(),
                ),
                'table_catalog': pa.array([CATALOG] * len(rows), pa.string()),
                'table_schema': pa.array([SCHEMA] * len(rows), pa.string()),
                'table_name': pa.array(
                    [table.lower()] * len(rows), pa.string()
                ),
                'table_type': pa.array(['MANAGED'] * len(rows), pa.string()),
            }
        )

    def _constraint_columns(
        self, table: str, kind: str
    ) -> list[tuple[str, str]]:
        rows = self._duck.execute(
            'SELECT constraint_type, constraint_column_names FROM '
            'duckdb_constraints() WHERE lower(table_name) = ? AND '
            'constraint_type = ?',
            [table.lower(), kind],
        ).fetchall()
        return [
            (f'{table.lower()}_{"_".join(cols)}_{kind.split()[0].lower()}', c)
            for _, cols in rows
            for c in cols
        ]

    def _keys(self, table: str) -> Any:
        import pyarrow as pa

        rows = [
            (kind, name, column)
            for kind in ('PRIMARY KEY', 'UNIQUE')
            for name, column in self._constraint_columns(table, kind)
        ]
        return pa.table(
            {
                'constraint_type': pa.array([r[0] for r in rows], pa.string()),
                'constraint_name': pa.array([r[1] for r in rows], pa.string()),
                'column_name': pa.array([r[2] for r in rows], pa.string()),
            }
        )

    def _foreign_keys(self, table: str) -> Any:
        import pyarrow as pa

        found = self._duck.execute(
            'SELECT constraint_column_names, referenced_table, '
            'referenced_column_names FROM duckdb_constraints() WHERE '
            "lower(table_name) = ? AND constraint_type = 'FOREIGN KEY'",
            [table.lower()],
        ).fetchall()
        rows = []
        for columns, target, targets in found:
            name = f'{table.lower()}_{"_".join(columns)}_foreign'
            for column, referenced in zip(columns, targets, strict=True):
                rows.append(
                    (name, column, CATALOG, SCHEMA, target.lower(), referenced)
                )
        names = (
            'constraint_name',
            'column_name',
            'table_catalog',
            'table_schema',
            'table_name',
            'column_name',
        )
        return pa.Table.from_arrays(
            [pa.array([r[i] for r in rows], pa.string()) for i in range(6)],
            names=list(names),
        )

    def _as_driver(self, table: Any) -> Any:
        r"""DuckDB's Arrow as ``fetchall_arrow`` gives it: plain strings, and
        a naive time in microseconds."""
        import pyarrow as pa

        fields = []
        for field in table.schema:
            kind = field.type
            if pa.types.is_large_string(kind) or pa.types.is_string_view(kind):
                kind = pa.string()
            if pa.types.is_timestamp(kind) and kind.unit != 'us':
                kind = pa.timestamp('us', tz=kind.tz)
            fields.append(pa.field(field.name, kind))
        return table.cast(pa.schema(fields))


# --- the runs ---------------------------------------------------------------------


def _bases() -> dict[str, list[str]]:
    import yaml

    return yaml.safe_load(BASES.read_text(encoding='utf-8'))['bases']


def _materialize_arguments(given: dict[str, Any]) -> dict[str, Any]:
    import pandas as pd

    arguments = dict(given)
    for name in ('anchor_time', 'context_anchor_time'):
        if name in arguments and arguments[name] != 'entity':
            arguments[name] = pd.Timestamp(arguments[name])
    if arguments.pop('indices_as', None) == 'decimal':
        arguments['indices'] = [
            decimal.Decimal(str(i)) for i in arguments['indices']
        ]
    return arguments


def run_case(
    path: str | dict[str, Any], bases: dict[str, list[str]]
) -> dict[str, Any]:
    import duckdb
    from sdm_client import PredictiveQuery, Relationship, Schema
    from sdm_connectors import DatabricksSource

    case = (
        json.loads((CASES / path).read_text(encoding='utf-8'))
        if isinstance(path, str)
        else path
    )
    given = case['input']
    duck = duckdb.connect()
    duck.execute("SET TimeZone = 'UTC'")
    for statement in bases[given['base']] + given.get('setup', []):
        duck.execute(statement)
    connection = Connection(duck)
    arguments = dict(given.get('from_source', {}))
    if 'relationships' in arguments:
        arguments['relationships'] = [
            Relationship(**r) for r in arguments['relationships']
        ]
    outcome = 'success'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            schema = Schema.from_source(
                DatabricksSource(connection, catalog=CATALOG, schema=SCHEMA),
                **arguments,
            )
            if case['until'] != 'from_source':
                PredictiveQuery(given['pql'], schema).materialize(
                    **_materialize_arguments(given['materialize'])
                )
        except Exception as error:  # noqa: BLE001 - the case's outcome
            outcome = type(error).__name__
    return {'case': case['id'], 'outcome': outcome, 'log': connection.log}


# --- writing ------------------------------------------------------------------------


def _rows(table: Any) -> list[tuple[Any, ...]]:
    r"""A table's rows, by position: a catalog answer may name two columns
    alike."""
    return (
        list(
            zip(*(column.to_pylist() for column in table.columns), strict=True)
        )
        if table.num_columns
        else []
    )


def _cell(value: Any) -> Any:
    if isinstance(value, decimal.Decimal):
        return str(value)
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, dict):
        return {k: _cell(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cell(v) for v in value]
    return value


def _result(table: Any, *, rows: bool) -> dict[str, Any]:
    r"""A statement's result: its schema, its Arrow IPC stream (zstd, base64)
    as the driver fetched it, and, for a catalog answer, its rows, to read."""
    import pyarrow as pa

    sink = pa.BufferOutputStream()
    options = pa.ipc.IpcWriteOptions(compression='zstd')
    with pa.ipc.new_stream(sink, table.schema, options=options) as writer:
        writer.write_table(table)
    result: dict[str, Any] = {
        'schema': [{'name': f.name, 'type': str(f.type)} for f in table.schema],
        'num_rows': table.num_rows,
    }
    if rows:
        result['rows'] = [[_cell(v) for v in row] for row in _rows(table)]
    result['arrow'] = base64.b64encode(sink.getvalue().to_pybytes()).decode(
        'ascii'
    )
    return result


def _shown(parameters: list[Any]) -> list[Any]:
    r"""Parameters as the reference statements show them: a long one (a key
    set's JSON) by its length and first characters."""
    shown = []
    for value in parameters:
        value = _cell(value)
        if isinstance(value, str) and len(value) > 120:
            value = {'json_prefix': value[:120], 'length': len(value)}
        shown.append(value)
    return shown


def _reading(
    text: str, parameters: list[Any], table: Any
) -> dict[str, Any] | None:
    r"""How the old client reads a catalog answer: its own code over the
    rows."""
    from sdm_relational_engine.rfm.backend.databricks.table import (
        DatabricksTable,
    )

    rows = _rows(table)
    if 'information_schema.columns' in text:
        return {
            'part': 'columns',
            'columns': [
                {
                    'name': name,
                    'declared_type': kind,
                    'not_null': nullable.strip().upper() == 'NO',
                    'dtype': None
                    if (d := DatabricksTable._to_dtype(kind)) is None
                    else str(d.value if hasattr(d, 'value') else d),
                }
                for name, kind, nullable, *_ in rows
            ],
            'canonical': list(rows[0][3:6]) if rows else None,
            'table_type': rows[0][6] if rows else None,
        }
    if "constraint_type IN ('PRIMARY KEY', 'UNIQUE')" in text:
        return {
            'part': 'keys',
            'keys': [
                {
                    'kind': kind.strip().upper(),
                    'name': name,
                    'columns': [column],
                }
                for kind, name, column in rows
            ],
        }
    if "constraint_type = 'FOREIGN KEY'" in text:
        return {
            'part': 'foreign_keys',
            'foreign_keys': [
                {
                    'name': r[0],
                    'columns': [r[1]],
                    'referenced_table': r[4],
                    'referenced_columns': [r[5]],
                }
                for r in rows
            ],
        }
    if 'information_schema.tables' in text:
        return {'part': 'tables', 'tables': [r[0] for r in rows]}
    return None


#: Each request kind of `DataRequest`, and how to tell the old client's
#: statement for it (its text) from the others.
KINDS = (
    (
        'describe/tables',
        lambda t: (
            'information_schema.tables' in t
            and 'information_schema.columns' not in t
        ),
    ),
    ('describe/columns', lambda t: 'information_schema.columns' in t),
    (
        'describe/keys',
        lambda t: "constraint_type IN ('PRIMARY KEY', 'UNIQUE')" in t,
    ),
    ('describe/foreign_keys', lambda t: "constraint_type = 'FOREIGN KEY'" in t),
    (
        'current_namespace',
        lambda t: t == 'SELECT current_catalog(), current_schema()',
    ),
    (
        'preview',
        lambda t: (
            re.fullmatch(r'SELECT .* FROM \S+ LIMIT \d+', t, re.S) is not None
        ),
    ),
    ('count_rows', lambda t: t.startswith('SELECT COUNT(*) FROM ')),
    ('key_counts', lambda t: t.startswith('SELECT 0 AS k, COUNT(*) AS n')),
    ('repeated_key', lambda t: 'HAVING COUNT(*) > 1' in t),
    ('column_ranges', lambda t: t.startswith('SELECT COUNT(*), MIN(')),
    ('time_bounds', lambda t: 'as table_index' in t),
    (
        'sample_entities',
        lambda t: 'TABLESAMPLE' in t or ('posexplode' in t and '__ids__' in t),
    ),
    (
        'lookup',
        lambda t: (
            'posexplode' in t
            and 'QUALIFY' not in t
            and '__KUMO_END_TIME__' not in t
            and '__ids__' not in t
        ),
    ),
    (
        'neighbors',
        lambda t: (
            'posexplode' in t and ('QUALIFY' in t or '__KUMO_END_TIME__' in t)
        ),
    ),
)


def kind_of(text: str) -> str:
    for name, test in KINDS:
        if test(text):
            return name
    return 'other'


#: Entity draws whose percentage and seed the old client writes as text
#: (RS-6, KO3-41): each (n, the table's rows, seed), the rows ``None`` where
#: the old client knows no count. The percentage is Python's ``str`` of
#: ``min(100.0, 100.0 * n / rows * 3.0)`` (DB/sampler.py:255-262): very
#: small (scientific notation), inexact, capped, whole, and the seeds 0,
#: negative and past 2**31.
DRAWS = (
    (10_000, None, 42),
    (1, 30_000_000, 42),
    (1, 1_000, 42),
    (5, 10, 42),
    (2, 100, 0),
    (2, 100, -1),
    (2, 100, 2**31 + 1),
    (7, 3, 42),
)


def sampling_vectors() -> list[dict[str, Any]]:
    r"""The old client's entity draw on ``users`` of the shop base for each
    of :data:`DRAWS`, its row count set as the sampler would have counted
    it: ``DatabricksSampler._sample_entity_table`` run on the stand-in
    connection, its statement recorded."""
    import duckdb
    from sdm_relational_engine.rfm import Graph
    from sdm_relational_engine.rfm.backend.databricks import DatabricksSampler

    duck = duckdb.connect()
    for statement in _bases()['shop']:
        duck.execute(statement)
    connection = Connection(duck)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        graph = Graph.from_databricks(
            connection, catalog=CATALOG, schema=SCHEMA, verbose=False
        )
        sampler = DatabricksSampler(graph, verbose=False)
    vectors = []
    for n, rows, seed in DRAWS:
        sampler._num_rows_dict['users'] = rows
        before = len(connection.log)
        sampler._sample_entity_table(
            'users', {'user_id'}, num_rows=n, random_seed=seed
        )
        (sent,) = connection.log[before:]
        vectors.append(
            {'n': n, 'rows': rows, 'seed': seed, 'text': sent['text']}
        )
    return vectors


def build() -> tuple[dict[str, Any], dict[str, Any]]:
    import duckdb
    import pyarrow
    import sdm_client

    bases = _bases()
    entries = []
    statements: dict[str, list[dict[str, Any]]] = {
        name: [] for name, _ in KINDS
    }
    statements['other'] = []
    for path in RECORDED:
        run = run_case(path, bases)
        recorded = []
        for item in run['log']:
            text, parameters, table = (
                item['text'],
                item['parameters'],
                item['result'],
            )
            entry = {
                'kind': kind_of(text),
                'text': text,
                'parameters': [_cell(p) for p in parameters],
                'result': _result(
                    table, rows=kind_of(text).startswith('describe/')
                ),
            }
            reading = _reading(text, parameters, table)
            if reading is not None:
                entry['reading'] = reading
            recorded.append(entry)
            seen = statements[entry['kind']]
            if all(s['text'] != text for s in seen):
                seen.append(
                    {
                        'case': run['case'],
                        'text': text,
                        'parameters': _shown(parameters),
                    }
                )
        if path not in STATEMENTS_ONLY:
            entries.append(
                {
                    'case': run['case'],
                    'leg': LEG,
                    'outcome': run['outcome'],
                    'statements': recorded,
                }
            )
    about = (
        "A fixture written by T3.4's test author before C-1, shaped as the "
        "design's sdm-capture/1 (choice 2), recorded by record.py beside it "
        'from the reference running each case on a stand-in Databricks SQL '
        'connection over DuckDB (no warehouse): every statement the old '
        "client sent with its parameters and result, and the old client's "
        "reading of each catalog answer. C-1's capture replaces it."
    )
    capture = {
        'format': 'sdm-capture/1',
        'fixture': True,
        'about': about,
        'recording_commit': None,
        'reference_commit': REFERENCE_COMMIT,
        'legs': {
            LEG: {
                'runtime': None,
                'python': sys.version.split()[0],
                'pandas': __import__('pandas').__version__,
                'driver': None,
                'pyarrow': pyarrow.__version__,
                'duckdb': duckdb.__version__,
                'sdm_client': getattr(sdm_client, '__version__', None),
            }
        },
        'placeholders': {'{catalog}': CATALOG, '{schema}': SCHEMA},
        'cases': entries,
    }
    reference = {
        'about': (
            "The old client's Databricks statements for each request kind of "
            'sdm_contract::data::DataRequest, as it sent them in the runs of '
            "capture.json (record.py): the reference side of T3.4's goldens "
            '(rust/crates/sdm-data/tests/databricks_goldens.rs), for the diff '
            "the T3.4 brief's exit 2 reviews. C-1's recorded statements "
            'replace them.'
        ),
        'reference_commit': REFERENCE_COMMIT,
        'kinds': statements,
        'sampling': sampling_vectors(),
    }
    return capture, reference


GOLDENS = (
    ROOT / 'rust' / 'crates' / 'sdm-data' / 'tests' / 'databricks_goldens.rs'
)


def _rust_text(literal: str) -> str:
    r"""A Rust string literal's text, for the escapes the goldens use."""
    out, at = [], 0
    while at < len(literal):
        char = literal[at]
        if char != '\\':
            out.append(char)
            at += 1
            continue
        code = literal[at + 1]
        if code == 'u':
            end = literal.index('}', at)
            out.append(chr(int(literal[at + 3 : end], 16)))
            at = end + 1
            continue
        out.append({'n': '\n', 't': '\t', '"': '"', '\\': '\\'}[code])
        at += 2
    return ''.join(out)


def _goldens() -> list[tuple[dict[str, Any], list[str]]]:
    source = GOLDENS.read_text(encoding='utf-8')
    found = []
    pattern = r'\(\s*r##"(.*?)"##,\s*&\[(.*?)\],?\s*\),'
    for request, body in re.findall(pattern, source, re.S):
        texts = re.findall(r'"((?:[^"\\]|\\.)*)"', body)
        found.append((json.loads(request), [_rust_text(t) for t in texts]))
    return found


def _kind(request: dict[str, Any]) -> str:
    if request['request'] == 'describe':
        return f'describe/{request["part"]}'
    return request['request']


def diff() -> int:
    import difflib

    reference = json.loads(
        (HERE / 'reference-statements.json').read_text(encoding='utf-8')
    )
    for request, statements in _goldens():
        kind = _kind(request)
        candidates = [s['text'] for s in reference['kinds'].get(kind, [])]
        if kind == 'sample_entities':
            candidates += [d['text'] for d in reference['sampling']]
        names = re.findall(
            r'"source":\s*\[[^\]]*"([^"]+)"\]', json.dumps(request)
        )
        named = [c for c in candidates if any(f'`{n}`' in c for n in names)]
        theirs = (named or candidates or [''])[0]
        ours = statements[-1] if statements else ''
        print(f'=== {kind}: {json.dumps(request, sort_keys=True)}')
        if any(
            ' '.join(ours.split()) == ' '.join(c.split()) for c in candidates
        ):
            print("(the old client's statement)")
            continue
        print(
            '\n'.join(
                difflib.unified_diff(
                    theirs.splitlines(),
                    ours.splitlines(),
                    'the old client',
                    'the core',
                    lineterm='',
                )
            )
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--diff', action='store_true')
    arguments = parser.parse_args()
    if arguments.diff:
        return diff()
    capture, reference = build()
    written = {
        HERE / 'capture.json': json.dumps(capture, indent=1, ensure_ascii=False)
        + '\n',
        HERE / 'reference-statements.json': json.dumps(
            reference, indent=1, ensure_ascii=False
        )
        + '\n',
    }
    stale = [
        p.name
        for p, text in written.items()
        if not p.exists() or p.read_text(encoding='utf-8') != text
    ]
    if arguments.check:
        if stale:
            print(f'stale: {", ".join(stale)}', file=sys.stderr)
            return 1
        print('up to date')
        return 0
    for path, text in written.items():
        path.write_text(text, encoding='utf-8')
    print(f'wrote {", ".join(p.name for p in written)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
