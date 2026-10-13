# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The canary test over the nested family's error paths (T1.32c's exit "the
canary test covers the family's error paths", T1.32a's exits applied to
this family; decision H17).

Written by the package's independent test author, before the
implementation. It drives each path of a nested column through the canary
of ``test_the_canary.py`` (its sentinels, its watch over the core's error,
``str``, ``repr``, the details, the traceback, every warning, every log
record at DEBUG, stderr and stdout, and its check that none of them holds a
sentinel), on the core route, ``Schema.from_source`` then ``materialize``:

- describe: a struct, a map, a list of lists, a list of structs and a
  fixed-size list holding the data sentinel (the reference fails each
  unnamed; the core refuses each by name, or describes it, as the package
  settles and registers it);
- describe: a list of text and a binary column holding the sentinel, which
  the reference leaves out with a warning that names the column, never a
  value;
- admission: a union holding the sentinel, and an interval holding our
  number sentinel, refused by name with the column and its type;
- materialize: a struct and a list on the fact table, which the request's
  context reads.

The sentinels are the canary's data sentinel and a number of our own,
``NUMBER``, carried by the nested column's values. Whatever the package
settles for a path, a refusal or a success, nothing written may hold one.
A refusal must be the core's (the canary saw the core's error), and never
a rule the core does not take yet (``UNSUPPORTED_REQUEST``: the brief lifts
``column_type`` for this family).

Gating (``contract/tests/t132c_gate.py``): a scenario named in
``t132c-pending.json``'s ``pending["exits"]`` (as ``canary:<scenario>``)
failed when written. It still runs and must still fail; once it passes it
fails, naming itself, until it is taken out of the list.
"""

from __future__ import annotations

import importlib.util
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pytest
from sdm_connectors import DuckDBSource

from sdm_client import PredictiveQuery, Schema, StructuredDataError, _routing

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def _module(name: str, path: Path) -> Any:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


C = _module('test_the_canary', HERE / 'test_the_canary.py')
G = _module('t132c_gate', ROOT / 'contract/tests/t132c_gate.py')
DATA = C.DATA
#: Our number sentinel: digits no statement, code or field holds by chance.
NUMBER = '73400917'
QUERY = 'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id'
CORE = ','.join(_routing.FROM_SOURCE_DUCKDB + _routing.MATERIALIZE_DUCKDB)


def _with(table: str, expression: str) -> list[str]:
    return [
        f'CREATE OR REPLACE TABLE {table} AS SELECT *, {expression} AS f '
        f'FROM {table}'
    ]


#: Each scenario: the table its column ``f`` is on, and the setup adding it.
SETUPS: dict[str, tuple[str, list[str]]] = {
    'describe: a struct holding the sentinel': (
        'users',
        _with('users', f"{{'a': user_id::INTEGER, 'b': '{DATA}'}}"),
    ),
    'describe: a map keyed by the sentinel': (
        'users',
        _with('users', f"map(['{DATA}'], [user_id::INTEGER])"),
    ),
    'describe: a list of lists holding the sentinel': (
        'users',
        _with('users', f"[['{DATA}']]::VARCHAR[][]"),
    ),
    'describe: a list of structs holding the sentinel': (
        'users',
        _with('users', f"[{{'a': '{DATA}'}}]"),
    ),
    'describe: a fixed-size list holding the sentinel': (
        'users',
        _with('users', f"['{DATA}', 'b']::VARCHAR[2]"),
    ),
    'describe: a list of text holding the sentinel': (
        'users',
        _with('users', f"['{DATA}']::VARCHAR[]"),
    ),
    'describe: a binary column holding the sentinel': (
        'users',
        _with('users', f"'{DATA}'::BLOB"),
    ),
    'admission: a union holding the sentinel': (
        'users',
        _with(
            'users', f"union_value(s := '{DATA}')::UNION(n INTEGER, s VARCHAR)"
        ),
    ),
    'admission: an interval holding the number': (
        'users',
        _with('users', f'to_days({NUMBER})'),
    ),
    'materialize: a struct on the fact table holding the sentinel': (
        'orders',
        _with('orders', f"{{'a': '{DATA}'}}"),
    ),
    'materialize: a list on the fact table holding the sentinel': (
        'orders',
        _with('orders', f"['{DATA}', '{NUMBER}']::VARCHAR[]"),
    ),
}


def _connection(setup: list[str]) -> Any:
    connection = duckdb.connect()
    for statement in C._shop() + setup:
        connection.execute(statement)
    return connection


def _materialize(setup: list[str]) -> Callable[[], Any]:
    def call() -> Any:
        schema = Schema.from_source(DuckDBSource(_connection(setup)))
        return PredictiveQuery(QUERY, schema).materialize(
            indices=[0, 1], anchor_time=pd.Timestamp('2024-03-01')
        )

    return call


def _answered(seen: Any, call: Callable[[], Any]) -> StructuredDataError | None:
    r"""The call's refusal, every text of it added to ``seen``, or None when
    it succeeded."""
    try:
        call()
    except StructuredDataError as error:
        seen.add('str', str(error))
        seen.add('repr', repr(error))
        seen.add('details', repr(error.details))
        seen.add('traceback', ''.join(traceback.format_exception(error)))
        return error
    return None


@pytest.fixture(autouse=True)
def _core_route(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    # Our number is a sentinel too, for this file's scenarios.
    monkeypatch.setattr(C, 'SENTINELS', (*C.SENTINELS, NUMBER))


@pytest.mark.parametrize('name', list(SETUPS))
def test_no_sentinel_of_the_nested_family_reaches_anything_written(
    name: str,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    def check() -> None:
        with C._watching(monkeypatch, capfd) as seen:
            error = _answered(seen, _materialize(SETUPS[name][1]))
        assert seen.leaks() == [], name
        if error is not None:
            # The core's error, never one the bridge makes in its place.
            assert any(where == 'core error' for where, _ in seen.texts), (
                f'{name}: refused outside the core ({error.code})'
            )
            assert error.code != 'UNSUPPORTED_REQUEST', (
                f'{name}: refused by a rule the core does not take yet'
            )

    G.held('exits', f'canary:{name}', check)


@pytest.mark.parametrize('name', list(SETUPS))
def test_each_sentinel_is_on_the_path(name: str) -> None:
    # The column the source hands over holds a sentinel, so the path that
    # reads it carries it.
    table, setup = SETUPS[name]
    connection = _connection(setup)
    cells = connection.execute(f'SELECT f::VARCHAR FROM {table}').fetchall()
    text = ' '.join(str(cell) for (cell,) in cells)
    if 'BLOB' in setup[0]:
        text = ' '.join(
            bytes(cell).decode()
            for (cell,) in connection.execute(
                f'SELECT f FROM {table}'
            ).fetchall()
        )
    assert DATA in text or NUMBER in text, (name, text[:200])


def test_the_watch_catches_each_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(C, 'SENTINELS', (*C.SENTINELS, NUMBER))
    seen = C._Seen()
    seen.add('warning', f'users leaves out f: {DATA}')
    seen.add('details', f'{{"value": "{NUMBER} days"}}')
    assert {sentinel for _, sentinel in seen.leaks()} == {NUMBER, DATA}
