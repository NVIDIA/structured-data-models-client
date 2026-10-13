# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The canary test over the empty, null and refuse family's error paths
(T1.32d's exit "the canary test covers the family's error paths", T1.32a's
exits applied to this family; decision H17: diagnostics carry no input
values).

Written by the package's independent test author, before the
implementation. It drives each path of the family through the canary of
``test_the_canary.py`` (its sentinels, its watch over the core's error,
``str``, ``repr``, the details, the traceback, every warning, every log
record at DEBUG, stderr and stdout, and its check that none of them holds a
sentinel), on the core route, ``Schema.from_source``, ``materialize`` and,
where the path goes on, ``SDMClient.predict`` against the scripted
endpoint. The data around each path holds the canary's data sentinel (two
users' countries) and a number sentinel of our own, ``NUMBER`` (an order's
price), so a message quoting the data it read holds one:

- describe: an empty fact table, an empty entity table and an empty third
  table (the reference fails each unnamed, ``Table '...' is empty``; the
  core refuses by name, rule ``empty_table``, P1B-11); a null key (the
  core refuses it by name, rule ``key_missing``);
- describe and materialize: an all-null feature, null foreign keys, null
  times, an all-null target (the reference warns that every context row
  has the same answer, naming no value), and an entity with no facts whose
  own key is the number;
- materialize: an empty entity list and entities the table does not hold,
  one of them the number (each refused with INVALID_REQUEST by the
  reference), and a question naming a column the schema does not hold;
- predict: a classification whose every context row answers the same
  (UNSUPPORTED_REQUEST at predict on the reference).

Whatever the package settles for a path, a refusal or a success, nothing
written may hold a sentinel. A refusal must be the core's (the canary saw
the core's error), except where the host refuses as the reference does:
the query's validation and the context judged before it is sent
(``HOST_MAY_REFUSE``). Which refusal it is, is not this test's to
judge: ``contract/tests/test_t132d_empty_null_vectors.py`` and
``t132d_empty_null_vectors.rs`` hold each path's outcome to the reference
and the register.

Gating (``contract/tests/t132d_gate.py``): a scenario named in
``t132d-pending.json``'s ``pending["exits"]`` (as ``canary:<scenario>``)
failed when written. It still runs and must still fail; once it passes it
fails, naming itself, until it is taken out of the list.
"""

from __future__ import annotations

import importlib.util
import sys
import traceback
import warnings
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
G = _module('t132d_gate', ROOT / 'contract/tests/t132d_gate.py')
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


#: Each scenario: its setup (after the planted sentinels), question,
#: entities, and whether it goes on to ``predict``.
SCENARIOS: dict[str, dict[str, Any]] = {
    'describe: an empty fact table': {'setup': ['DELETE FROM orders']},
    'describe: an empty entity table': {
        'setup': ['DELETE FROM users'],
    },
    'describe: an empty third table': {
        'setup': ['CREATE TABLE extra (extra_id BIGINT, v INTEGER)'],
    },
    'describe: a null key': {
        'setup': ['UPDATE users SET user_id = NULL WHERE user_id = 3'],
    },
    'materialize: an all-null feature': {
        'setup': [
            'CREATE OR REPLACE TABLE users AS SELECT *, NULL::INTEGER AS f '
            'FROM users'
        ],
        'predict': True,
    },
    'materialize: null foreign keys': {
        'setup': ['UPDATE orders SET user_id = NULL WHERE order_id % 10 = 1'],
        'predict': True,
    },
    'materialize: null times': {
        'setup': ['UPDATE orders SET ts = NULL WHERE order_id % 10 = 1'],
        'predict': True,
    },
    'materialize: an all-null target': {
        'setup': ['UPDATE orders SET price = NULL'],
        'predict': True,
    },
    'materialize: an entity with no facts keyed by the number': {
        'setup': [f"INSERT INTO users VALUES ({NUMBER}, 50, '{DATA}')"],
        'indices': [0, int(NUMBER)],
        'predict': True,
    },
    'materialize: an empty entity list': {'indices': []},
    'materialize: an entity the table does not hold, the number': {
        'indices': [0, int(NUMBER)],
    },
    'materialize: a column the schema does not hold': {
        'query': 'PREDICT SUM(orders.cost, 0, 30, days) FOR EACH users.user_id',
    },
    'predict: one class in the context': {
        'query': (
            'PREDICT COUNT(orders.*, 0, 30, days) > 1000 FOR EACH users.user_id'
        ),
        'predict': True,
    },
}


#: The paths the host refuses on the core route, as the reference does:
#: the query's validation (``contract/refusal-maps/refusal.json``'s entry
#: ``materialize/unknown_column``: python-core validates the query in the
#: host) and the context the host judges before it sends. Every other
#: refusal must be the core's.
HOST_MAY_REFUSE = {
    'materialize: a column the schema does not hold',
    'predict: one class in the context',
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
            indices=list(scenario.get('indices', [0, 1])),
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
    # Our number is a sentinel too, for this file's scenarios.
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
        if error is not None and name not in HOST_MAY_REFUSE:
            # The core's error, never one the bridge makes in its place.
            assert any(where == 'core error' for where, _ in seen.texts), (
                f'{name}: refused outside the core ({error.code})'
            )

    G.held('exits', f'canary:{name}', check)


def test_each_sentinel_is_on_the_path() -> None:
    # The data the paths read holds both sentinels: the source hands them
    # over, and a request of the family sends them.
    connection = _connection([])
    countries = {
        c for (c,) in connection.execute('SELECT country FROM users').fetchall()
    }
    prices = {
        str(int(p))
        for (p,) in connection.execute('SELECT price FROM orders').fetchall()
    }
    assert DATA in countries and NUMBER in prices
    with C._endpoint().serve() as (url, received), warnings.catch_warnings():
        # The journey's own warnings, which the scenarios watch.
        warnings.simplefilter('ignore')
        schema = Schema.from_source(DuckDBSource(_connection([])))
        request = PredictiveQuery(QUERY, schema).materialize(
            indices=[3, 5], anchor_time=pd.Timestamp('2024-03-01')
        )
        with SDMClient(url, max_retries=0) as client:
            client.predict(request, model=KumoRelational())
    [(_, _, body)] = received.requests
    assert DATA.encode() in body


def test_the_watch_catches_each_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(C, 'SENTINELS', (*C.SENTINELS, NUMBER))
    seen = C._Seen()
    seen.add('warning', f'every context row has the same {DATA}')
    seen.add('details', f'{{"indices": [{NUMBER}]}}')
    assert {sentinel for _, sentinel in seen.leaks()} == {NUMBER, DATA}
