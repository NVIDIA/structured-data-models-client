# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P3.seam: the engine reaches a database only through ``Source._run``.

The engine's source is parsed, not searched as text, so a name in a string or
a comment is not a call, and a call split over lines, made through
``getattr``, or a driver method handed on as a value is still found. Each of
these names is a driver call the connectors now make for the engine:
``cursor``, ``execute``, ``executemany``, ``fetchall``, ``fetchone``,
``fetchmany``, ``fetchall_arrow``, ``fetch_arrow_all``, ``fetch_arrow_table``,
``to_arrow_table``, ``toArrow``, ``register``, ``adbc_ingest`` and
``commit``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from _seam import DRIVER_CALLS, ENGINE_SRC, REPO


def driver_calls(source: str, filename: str = '<source>') -> list[str]:
    r"""Each place ``source`` calls, fetches or references a driver method,
    as ``file:line: what``."""
    tree = ast.parse(source, filename=filename)
    found: list[str] = []
    called: set[int] = set()

    def note(node: ast.AST, what: str) -> None:
        found.append(f'{filename}:{getattr(node, "lineno", 0)}: {what}')

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in DRIVER_CALLS:
            called.add(id(func))
            note(node, f'call of .{func.attr}()')
        elif isinstance(func, ast.Name) and func.id in DRIVER_CALLS:
            note(node, f'call of {func.id}()')
        elif (
            isinstance(func, ast.Name)
            and func.id in ('getattr', 'hasattr')
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in DRIVER_CALLS
        ):
            note(node, f'{func.id}(..., {node.args[1].value!r})')
        elif (
            isinstance(func, ast.Attribute)
            and func.attr in ('methodcaller', 'attrgetter')
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value in DRIVER_CALLS
        ):
            note(node, f'{func.attr}({node.args[0].value!r})')
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr in DRIVER_CALLS
            and isinstance(node.ctx, ast.Load)
            and id(node) not in called
        ):
            note(node, f'reference to .{node.attr}')
    return sorted(found)


def engine_files() -> list[Path]:
    return sorted(ENGINE_SRC.rglob('*.py'))


def test_the_engine_source_is_scanned() -> None:
    files = engine_files()
    names = {p.relative_to(ENGINE_SRC).as_posix() for p in files}
    assert len(files) >= 50, f'only {len(files)} engine files found'
    for expected in (
        'sdm_relational_engine/rfm/graph.py',
        'sdm_relational_engine/rfm/base/sql_sampler.py',
    ):
        assert expected in names, f'{expected} is not scanned'


def test_no_engine_module_calls_a_driver() -> None:
    found: list[str] = []
    for path in engine_files():
        rel = path.relative_to(REPO).as_posix()
        found += driver_calls(path.read_text(encoding='utf-8'), rel)
    assert not found, (
        f'{len(found)} driver call(s) left in the engine; each should be a '
        f'Source._run:\n  ' + '\n  '.join(found)
    )


# The checker itself: what it must find, and what it must not.

_FOUND = {
    'a cursor': 'with connection.cursor() as cursor:\n    pass\n',
    'an execute': 'cursor.execute(sql, parameters=[1])\n',
    'a split call': 'cursor.execute(\n    sql,\n)\n',
    'a chained fetch': 'rows = connection.cursor().execute(sql).fetchall()\n',
    'executemany': 'cursor.executemany(sql, rows)\n',
    'fetchone': 'row = cursor.fetchone()\n',
    'fetchmany': 'rows = cursor.fetchmany(10)\n',
    'Databricks Arrow': 'table = cursor.fetchall_arrow()\n',
    'Snowflake Arrow': 'table = cursor.fetch_arrow_all(force_return_table=True)\n',
    'ADBC Arrow': 'table = cursor.fetch_arrow_table()\n',
    'DuckDB Arrow': 'table = relation.to_arrow_table()\n',
    'Spark Arrow': 'table = frame.toArrow()\n',
    'register': 'connection.register(name, frame)\n',
    'adbc_ingest': "cursor.adbc_ingest('t', data, mode='replace')\n",
    'commit': 'connection.commit()\n',
    'getattr': "fetch = getattr(cursor, 'fetchall')\n",
    'a bound method': 'self._raise_timeout(cursor.execute)\n',
    'methodcaller': "call = operator.methodcaller('execute', sql)\n",
}

_NOT_FOUND = {
    'a string': "sql = 'cursor.execute(x); connection.commit()'\n",
    'a comment': '# cursor.execute(sql) then fetchall()\nx = 1\n',
    'a docstring': 'def f():\n    """Calls cursor.execute()."""\n',
    'a definition': 'def execute(self, sql):\n    return sql\n',
    'another name': (
        'source._run(sql, params)\nrun(connection, sql, params)\n'
        'rows.fetch_all_rows()\n'
    ),
}


@pytest.mark.parametrize('label', sorted(_FOUND))
def test_the_check_finds(label: str) -> None:
    assert driver_calls(_FOUND[label]), f'{label} was not found'


@pytest.mark.parametrize('label', sorted(_NOT_FOUND))
def test_the_check_ignores(label: str) -> None:
    assert not driver_calls(_NOT_FOUND[label]), f'{label} was reported'
