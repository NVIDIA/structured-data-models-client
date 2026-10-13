# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The canary test over the time and feature family's error paths and the
multicategorical port's (T1.32e part 2's exit "the canary test covers the
family's error paths", T1.32a's exits applied to this family; decision
H17: diagnostics carry no input values).

Written by the package's independent test author, before the
implementation. It drives each path through the canary of
``test_the_canary.py`` (its sentinels, its watch over the core's error,
``str``, ``repr``, the details, the traceback, every warning, every log
record at DEBUG, stderr and stdout, and its check that none of them holds a
sentinel), on the core route, ``Schema.from_source``, ``materialize`` and,
where the path goes on, ``SDMClient.predict`` against the scripted
endpoint. The data around each path holds the canary's data sentinel (two
users' countries) and a number sentinel of our own, ``NUMBER`` (an order's
price); where the path reads a column of its own, the column holds one too:

- describe: text holding several categories per value, one of them the
  data sentinel, on the entity table and on the fact table, a null and an
  empty string beside them (the reference describes it as
  ``multicategorical``; the core refuses it today, rule
  ``multicategorical``, which the port lifts); text holding times of day,
  one value the data sentinel; an interval holding the number (refused by
  name, P1B-8); a date and a timestamp, one with nulls, the earlier chosen
  with a warning;
- materialize: text dates the source cannot cast, one value the data
  sentinel, one the number (the reference fails INTERNAL_ERROR in the
  source's CAST; the core's provider error, DT-12, must not quote what the
  source said of the value); a time of day as a feature (the engine warns
  of the values it drops);
- materialize and predict: a multi-category feature reaching the request;
  a multi-category column asked for as the target, and summed (the
  reference refuses both at materialize).

The paths the port opens (``PORTED``) must reach past the description:
a refusal by the rule ``multicategorical`` fails them, so the canary
watches the port's path. Whatever the package settles for a path, a
refusal or a success, nothing written may hold a sentinel. A refusal must be the core's (the canary saw
the core's error), except where the host refuses as the reference does:
the query's validation (``HOST_MAY_REFUSE``). Which refusal it is, is not
this test's to judge: ``contract/tests/test_t132e_multicategorical.py``,
``t132e_multicategorical.rs`` and the family's tests hold each outcome to
the reference and the register.

Gating (``contract/tests/t132e_gate.py``): a scenario named in
``t132e-pending.json``'s ``pending["p2-exits"]`` (as ``canary:<scenario>``)
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

from sdm_client import (
    KumoRelational,
    PredictiveQuery,
    Schema,
    SDMClient,
    StructuredDataError,
    _routing,
)

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
G = _module('t132e_gate', ROOT / 'contract/tests/t132e_gate.py')
DATA = C.DATA
#: Our number sentinel: digits no statement, code or field holds by chance.
NUMBER = '73400917'
QUERY = 'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id'
CORE = ','.join(
    _routing.FROM_SOURCE_DUCKDB
    + _routing.MATERIALIZE_DUCKDB
    + _routing.PREDICT_DUCKDB
)
#: The sentinels planted around every scenario: two users' countries and
#: an order's price.
PLANTED = [
    f"UPDATE users SET country = '{DATA}' WHERE user_id IN (3, 5)",
    f'UPDATE orders SET price = {NUMBER} WHERE order_id = 11',
]


def _with(table: str, expression: str) -> list[str]:
    return [
        f'CREATE OR REPLACE TABLE {table} AS SELECT *, {expression} AS f '
        f'FROM {table}'
    ]


def _cycle(values: list[str | None], over: str) -> str:
    items = ', '.join('NULL' if v is None else f"'{v}'" for v in values)
    return f'([{items}]::VARCHAR[])[1 + {over} % {len(values)}]'


#: Multi-category values, one part the data sentinel, a null and an empty
#: string among them (eight distinct values of four parts and more: the
#: reference's ratio holds).
MULTI = ['a;b', f'a;{DATA}', 'b;c', 'a;b;c', 'a', None, '', f'c;{DATA};a']

#: Each scenario: its setup (after the planted sentinels), the question
#: and entities, and whether it goes on to ``predict``.
SCENARIOS: dict[str, dict[str, Any]] = {
    'describe: multi-category text holding the sentinel': {
        'setup': _with('users', _cycle(MULTI, 'user_id')),
        'predict': True,
    },
    'describe: multi-category text on the fact table holding the sentinel': {
        'setup': _with('orders', _cycle(MULTI, 'order_id')),
        'predict': True,
    },
    'describe: times of day as text, one the sentinel': {
        'setup': _with(
            'users',
            _cycle(['09:30', '10:15', DATA, '17:45', '08:00'], 'user_id'),
        ),
        'predict': True,
    },
    'describe: an interval holding the number': {
        'setup': _with('users', f'to_days({NUMBER})'),
    },
    'describe: a date and a timestamp, one with nulls': {
        'setup': [
            'CREATE OR REPLACE TABLE users AS SELECT *, '
            "DATE '2020-01-01' + to_days(user_id::INTEGER) AS f_date, "
            "CASE WHEN user_id = 3 THEN NULL ELSE TIMESTAMP '2021-01-01' "
            '+ to_days(user_id::INTEGER) END AS f_ts FROM users'
        ],
        'predict': True,
    },
    'materialize: text dates the source cannot cast, one the sentinel': {
        'setup': [
            'ALTER TABLE orders ALTER ts TYPE VARCHAR',
            f"UPDATE orders SET ts = '{DATA}' WHERE order_id = 40",
        ],
        'predict': True,
    },
    'materialize: text dates the source cannot cast, one the number': {
        'setup': [
            'ALTER TABLE orders ALTER ts TYPE VARCHAR',
            f"UPDATE orders SET ts = '{NUMBER}' WHERE order_id = 40",
        ],
        'predict': True,
    },
    'materialize: a time of day as a feature': {
        'setup': _with('users', 'make_time(user_id::BIGINT, 30, 0)'),
        'predict': True,
    },
    'materialize: a multi-category target': {
        'setup': _with('users', _cycle(MULTI, 'user_id')),
        'query': 'PREDICT users.f FOR EACH users.user_id',
        'predict': True,
    },
    'materialize: a multi-category column summed': {
        'setup': _with('orders', _cycle(MULTI, 'order_id')),
        'query': 'PREDICT SUM(orders.f, 0, 30, days) FOR EACH users.user_id',
        'predict': True,
    },
}


#: The paths the host may refuse on the core route, as the reference
#: does: the query's validation (python-core validates the query in the
#: host before the core's materialize). Every other refusal must be the
#: core's.
HOST_MAY_REFUSE = {
    'materialize: a multi-category target',
    'materialize: a multi-category column summed',
}


#: The paths the multicategorical port opens: each must reach past the
#: description (no refusal by the rule ``multicategorical``), so the canary
#: watches the path the port takes, not today's refusal in its place.
PORTED = {
    'describe: multi-category text holding the sentinel',
    'describe: multi-category text on the fact table holding the sentinel',
    'describe: times of day as text, one the sentinel',
    'materialize: a multi-category target',
    'materialize: a multi-category column summed',
}


def _connection(setup: list[str]) -> Any:
    connection = duckdb.connect()
    for statement in C._shop() + PLANTED + setup:
        connection.execute(statement)
    return connection


def _call(scenario: dict[str, Any]) -> Callable[[], Any]:
    def call() -> Any:
        schema = Schema.from_source(
            DuckDBSource(_connection(scenario.get('setup', [])))
        )
        request = PredictiveQuery(
            scenario.get('query', QUERY), schema
        ).materialize(
            indices=[0, 1, 3, 5],
            anchor_time=pd.Timestamp('2024-03-01'),
        )
        if not scenario.get('predict'):
            return request
        with C._endpoint().serve() as (url, _):
            with SDMClient(url, max_retries=0) as client:
                return client.predict(request, model=KumoRelational())

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
    monkeypatch.setattr(C, 'SENTINELS', (*C.SENTINELS, NUMBER))


@pytest.mark.parametrize('name', list(SCENARIOS))
def test_no_sentinel_of_the_family_reaches_anything_written(
    name: str,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    def check() -> None:
        with C._watching(monkeypatch, capfd) as seen:
            error = _answered(seen, _call(SCENARIOS[name]))
        assert seen.leaks() == [], name
        if name in PORTED and error is not None:
            details = error.details if isinstance(error.details, dict) else {}
            assert details.get('rule') != 'multicategorical', (
                f'{name}: still refused for holding a separator'
            )
        if error is not None and name not in HOST_MAY_REFUSE:
            # The core's error, never one the bridge makes in its place.
            assert any(where == 'core error' for where, _ in seen.texts), (
                f'{name}: refused outside the core ({error.code})'
            )

    G.held('p2-exits', f'canary:{name}', check)


@pytest.mark.parametrize(
    'name',
    [
        'describe: multi-category text holding the sentinel',
        'describe: multi-category text on the fact table holding the sentinel',
        'describe: times of day as text, one the sentinel',
        'materialize: text dates the source cannot cast, one the sentinel',
    ],
)
def test_each_sentinel_is_on_the_path(name: str) -> None:
    # The column the path reads holds the sentinel, so a message quoting
    # what it read would hold it.
    connection = _connection(SCENARIOS[name]['setup'])
    table = 'orders' if 'fact table' in name or 'dates' in name else 'users'
    column = 'ts' if 'dates' in name else 'f'
    values = {
        str(v)
        for (v,) in connection.execute(
            f'SELECT "{column}" FROM "{table}"'
        ).fetchall()
    }
    assert any(DATA in v for v in values), name


def test_the_number_is_on_the_path() -> None:
    connection = _connection(
        SCENARIOS[
            'materialize: text dates the source cannot cast, one the number'
        ]['setup']
    )
    values = {
        v for (v,) in connection.execute('SELECT ts FROM orders').fetchall()
    }
    assert NUMBER in values
    prices = {
        str(int(p))
        for (p,) in connection.execute('SELECT price FROM orders').fetchall()
    }
    assert NUMBER in prices


def test_the_watch_catches_each_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(C, 'SENTINELS', (*C.SENTINELS, NUMBER))
    seen = C._Seen()
    seen.add('core error', f'invalid timestamp field format: "{DATA}"')
    seen.add('warning', f'the parts of f include {NUMBER}')
    assert {sentinel for _, sentinel in seen.leaks()} == {NUMBER, DATA}
