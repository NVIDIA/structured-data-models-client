# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.2: a row read's types in its database's driver's vocabulary.

The reference reads SQLite's catalog as rows through ADBC, whose cursor
names each column's type by its Arrow name (``int64``, ``string``), where
DuckDB's names its own types (``BIGINT``). ``compare.py`` holds a core
runner's Arrow schema to a row read's types through the database's
vocabulary (:func:`compare.row_arrow_name`): equal types pass, a retyped
column and a name outside the vocabulary fail, on SQLite as on DuckDB.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = next((ROOT / 'contract/reference').glob('*.jsonl'))


def _module(name: str, path: Path) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


compare = _module('compare', ROOT / 'scripts/rust_core/compare.py')


def _table_info() -> dict[str, Any]:
    r"""python-ref's ``PRAGMA table_info`` of ``orders`` on SQLite, a row
    read."""
    for line in REFERENCE.read_text().splitlines():
        observation = json.loads(line)
        if observation['case'] == 'base.regression@sqlite':
            for statement in observation['driver']['capture']['statements']:
                if statement['sql'] == 'PRAGMA table_info("orders")':
                    return statement
    raise AssertionError('no reference line holds the statement')


def _arrow_read(statement: dict[str, Any]) -> dict[str, Any]:
    r"""The same result read as Arrow by a core runner, its schema the row
    read's types."""
    read = statement['results'][0]
    schema = [
        {'name': name, 'type': kind}
        for name, kind in zip(
            read['values']['columns'], read['values']['types'], strict=True
        )
    ]
    return dict(
        statement,
        results=[
            {
                'method': 'query_arrow',
                'rows': read['rows'],
                'schema': schema,
                'values': dict(read['values'], types=None),
            }
        ],
    )


def _failures(
    ours: dict[str, Any], theirs: dict[str, Any], database: str
) -> list[str]:
    report = compare.Report({'sql.result': 'gate'})
    compare._results(
        report,
        'at',
        ours,
        theirs,
        False,
        ('rust-host', 'python-ref'),
        database=database,
    )
    return report.texts


def test_a_sqlite_row_read_s_arrow_names_are_held_to_the_schema() -> None:
    statement = _table_info()
    assert statement['results'][0]['method'] == 'fetchall'
    assert statement['results'][0]['values']['types'][0] == 'int64'
    arrow = _arrow_read(statement)
    assert _failures(arrow, statement, 'sqlite') == []
    # Read in DuckDB's vocabulary, the same names are no type of its.
    assert any(
        'no Arrow name' in t for t in _failures(arrow, statement, 'duckdb')
    )
    retyped = copy.deepcopy(arrow)
    retyped['results'][0]['schema'][0]['type'] = 'int32'
    assert any(
        'differs from the types' in t
        for t in _failures(retyped, statement, 'sqlite')
    )
    # A name outside ADBC SQLite's vocabulary is never taken as equal.
    foreign = copy.deepcopy(statement)
    foreign['results'][0]['values']['types'][0] = 'INTEGER'
    assert any(
        'no Arrow name' in t for t in _failures(arrow, foreign, 'sqlite')
    )
