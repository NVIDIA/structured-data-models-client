#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Presentation types: independent inputs and current-client test helpers."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'contract/result-conversion-vectors.json'
FORMAT = 'sdm-l5-vectors/1'

#: The L5 table, built on DuckDB: the test's ``SETUP``, which the test holds
#: equal to this.
SETUP = [
    'CREATE TABLE t (id BIGINT PRIMARY KEY, b BOOLEAN, i8 TINYINT, '
    'i16 SMALLINT, i32 INTEGER, i64 BIGINT, u8 UTINYINT, u16 USMALLINT, '
    'u32 UINTEGER, u64 UBIGINT, f32 FLOAT, f64 DOUBLE, dec DECIMAL(38,10), '
    's VARCHAR, bin BLOB, d DATE, tm TIME, ts TIMESTAMP, tstz TIMESTAMPTZ, '
    'iv INTERVAL, li INTEGER[], st STRUCT(a INTEGER, b VARCHAR), '
    'mp MAP(VARCHAR, INTEGER), un UNION(n INTEGER, s VARCHAR))',
    'INSERT INTO t VALUES (1, true, -128, -32768, -2147483648, '
    '-9223372036854775808, 0, 0, 0, 0, 1.5, 0.1, '
    "12345678901234567890.0123456789, 'a', '\\x00\\x01'::BLOB, "
    "'1970-01-01', '00:00:00', '2024-01-02 03:04:05.123456', "
    "'2024-01-02 03:04:05.123456+00', INTERVAL 1 DAY, [1, 2], "
    "{'a': 1, 'b': 'x'}, MAP {'k': 1}, 2::UNION(n INTEGER, s VARCHAR))",
    'INSERT INTO t VALUES (2, false, 127, 32767, 2147483647, '
    '9223372036854775807, 255, 65535, 4294967295, 18446744073709551615, '
    "-0.25, -1e300, -0.5, 'é ü', ''::BLOB, '2024-02-29', "
    "'23:59:59.123456', '1969-12-31 23:59:59.999999', "
    "'1999-12-31 23:00:00+00', INTERVAL '1 month 2 days 3 seconds', [], "
    "{'a': NULL, 'b': NULL}, MAP {}, 'z'::UNION(n INTEGER, s VARCHAR))",
    'INSERT INTO t (id) VALUES (3)',
]
COLUMNS = [
    'b', 'i8', 'i16', 'i32', 'i64', 'u8', 'u16', 'u32', 'u64', 'f32', 'f64',
    'dec', 's', 'bin', 'd', 'tm', 'ts', 'tstz', 'iv', 'li', 'st', 'mp', 'un',
]  # fmt: skip

#: Each layout's answer, as the endpoint answers it: the test's ``LAYOUTS``'
#: answered items, which the test holds equal to these.
ANSWERED: dict[str, list[dict[str, Any]]] = {
    'regression, float answers': [
        {'row_index': 0, 'id': 7, 'prediction': 1.5},
        {'row_index': 1, 'id': 8, 'prediction': None},
        {'row_index': 2, 'id': None, 'prediction': -2.25},
    ],
    'regression, integer answers': [
        {'row_index': 0, 'id': 1, 'prediction': 3},
        {'row_index': 1, 'id': 2, 'prediction': -9007199254740993},
    ],
    'regression, an integer widened among floats': [
        {'row_index': 0, 'id': 1, 'prediction': 1},
        {'row_index': 1, 'id': 2, 'prediction': 2.5},
        {'row_index': 2, 'id': 3, 'prediction': None},
    ],
    'binary classification, text ids': [
        {
            'row_index': 0,
            'id': 'a',
            'prediction': True,
            'probabilities': {'True': 0.75, 'False': 0.25},
        },
        {
            'row_index': 1,
            'id': None,
            'prediction': False,
            'probabilities': {'True': 0.0},
        },
        {'row_index': 2, 'id': 'é', 'prediction': None},
    ],
    'multiclass, ids of mixed types': [
        {'row_index': 0, 'id': 7, 'prediction': 'red'},
        {'row_index': 1, 'id': 'b', 'prediction': None},
        {'row_index': 2, 'id': None, 'prediction': 'blue'},
    ],
    'forecast, a step per row': [
        {'row_index': 0, 'id': 5, 'forecast_step': 1, 'prediction': 0.5},
        {'row_index': 0, 'id': 5, 'forecast_step': 2, 'prediction': 0.25},
        {'row_index': 1, 'id': 6, 'prediction': 1e300},
    ],
}


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def arrow_type(field: Any) -> str:
    r"""``field``'s type as the test spells it (``_arrow_type``): an
    extension type as its storage type and name."""
    import pyarrow as pa

    kind = field.type
    name = None
    if isinstance(kind, pa.BaseExtensionType):
        name = kind.extension_name
        kind = kind.storage_type
    elif field.metadata and b'ARROW:extension:name' in field.metadata:
        name = field.metadata[b'ARROW:extension:name'].decode()
    return str(kind) + (f' {name}' if name else '')


def presentation() -> dict[str, dict[str, str]]:
    r"""Each layout's presented columns and dtypes, by this interpreter's
    client."""
    from sdm_api.generated.tfm_api import PredictionItem
    from sdm_client.result import PredictionResult

    out = {}
    for layout, items in ANSWERED.items():
        frame = PredictionResult(
            model='m',
            request_id='r',
            items=[PredictionItem(**item) for item in items],
        ).to_pandas()
        out[layout] = {name: str(kind) for name, kind in frame.dtypes.items()}
    return out


def drivers() -> dict[str, dict[str, str]]:
    r"""Each column's Arrow type as the reference's own reads give it on
    DuckDB and SQLite."""
    import adbc_driver_sqlite.dbapi
    import duckdb

    loader = _module('databases', ROOT / 'contract/runners/databases.py')
    duck = duckdb.connect()
    # The test's pinned zone: a TIMESTAMPTZ's Arrow zone is the session's.
    duck.execute("SET GLOBAL TimeZone = 'Etc/UTC'")
    out: dict[str, dict[str, str]] = {}
    try:
        for statement in SETUP:
            duck.execute(statement)
        with tempfile.TemporaryDirectory(prefix='l5-record-') as tmp:
            path = Path(tmp) / 'l5.sqlite'
            loader.load(duck, 'sqlite', path)
            sqlite = adbc_driver_sqlite.dbapi.connect(str(path))
            try:
                for column in COLUMNS:
                    sql = f'SELECT "id", "{column}" FROM t ORDER BY id'
                    with duck.cursor() as cursor:
                        cursor.execute(sql)
                        on_duckdb = cursor.to_arrow_table()
                    with sqlite.cursor() as cursor:
                        cursor.execute(sql)
                        on_sqlite = cursor.fetch_arrow_table()
                    out[column] = {
                        'duckdb': arrow_type(on_duckdb.schema.field(column)),
                        'sqlite': arrow_type(on_sqlite.schema.field(column)),
                    }
            finally:
                sqlite.close()
    finally:
        duck.close()
    return out
