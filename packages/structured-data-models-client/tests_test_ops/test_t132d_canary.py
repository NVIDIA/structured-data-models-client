# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native value conversion must not expose planted data through errors or logs."""

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
DATA = C.DATA
#: Our number sentinel: digits no statement, code or field holds by chance.
NUMBER = '73400917'
QUERY = 'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id'
#: The entities of the scenarios that name one past the shop's keys,
#: holding no sentinel (an error may name the caller's own, KO2-86).
NO_FACTS = 999
NOT_HELD = 99
#: The sentinels planted around every scenario: two users' countries and
#: an order's price.
PLANTED = [
    f"UPDATE users SET country = '{DATA}' WHERE user_id IN (3, 5)",
    f'UPDATE orders SET price = {NUMBER} WHERE order_id = 11',
]


#: Each scenario: its setup (after the planted sentinels), the keyword
#: arguments of ``Schema.from_source``, question, entities, and whether it
#: goes on to ``predict``.
SCENARIOS: dict[str, dict[str, Any]] = {
    'describe: an empty fact table': {'setup': ['DELETE FROM orders']},
    'describe: an empty entity table': {
        'setup': ['DELETE FROM users'],
    },
    'describe: an empty third table': {
        'setup': ['CREATE TABLE extra (extra_id BIGINT, v INTEGER)'],
    },
    # T1.32e: a null in an inferred key no longer reaches a core refusal
    # (the key is declined below 99% distinct, as the reference declines
    # it), so the core's error path is reached by naming the key.
    'describe: a null key': {
        'setup': ['UPDATE users SET user_id = NULL WHERE user_id = 3'],
        'from_source': {'primary_keys': {'users': 'user_id'}},
    },
    'describe: a null key, inferred': {
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
    # From KO2-86 an error may name the entities the caller passed, so they
    # hold no sentinel; what the data holds stays out.
    'materialize: an entity with no facts': {
        'setup': [f"INSERT INTO users VALUES ({NO_FACTS}, 50, '{DATA}')"],
        'indices': [0, NO_FACTS],
        'predict': True,
    },
    'materialize: an empty entity list': {'indices': []},
    'materialize: an entity the table does not hold': {
        'indices': [0, NOT_HELD],
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
#: query validation, an entity column left without a key by a null,
#: and context validation before sending. Every other refusal must come
#: from the core.
HOST_MAY_REFUSE = {
    'describe: a null key, inferred',
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
            DuckDBSource(_connection(scenario.get('setup', []))),
            **scenario.get('from_source', {}),
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

    check()


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
    # No entity a scenario passes holds a sentinel (KO2-86).
    for scenario in SCENARIOS.values():
        for given in scenario.get('indices', []):
            assert DATA not in str(given) and NUMBER not in str(given)
