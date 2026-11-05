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
#: The data sentinel planted around every scenario: two users' countries.
PLANTED = [f"UPDATE users SET country = '{DATA}' WHERE user_id IN (3, 5)"]


def _two_orders(sql_type: str, a: str, b: str) -> list[str]:
    r"""Every price zero in ``sql_type``, and two orders of user 0 in one
    label window, priced ``a`` and ``b``."""
    return [
        f'ALTER TABLE orders ALTER price TYPE {sql_type} USING 0',
        f"INSERT INTO orders VALUES (1000, 0, '{a}'::{sql_type}, "
        "TIMESTAMP '2024-02-10')",
        f"INSERT INTO orders VALUES (1001, 0, '{b}'::{sql_type}, "
        "TIMESTAMP '2024-02-11')",
    ]


def _users_column(name: str, sql_type: str, values: list[str]) -> list[str]:
    whens = ' '.join(
        f"WHEN {i} THEN '{v}'::{sql_type}" for i, v in enumerate(values)
    )
    return [
        f'CREATE OR REPLACE TABLE users AS SELECT *, '
        f'CASE user_id % {len(values)} {whens} END AS {name} FROM users'
    ]


#: Each scenario: its setup (after the planted sentinel), question,
#: entities, anchor time (None: none), the tables' time columns, and
#: whether it goes on to ``predict``.
SCENARIOS: dict[str, dict[str, Any]] = {
    'materialize: a sum past the declared precision': {
        'setup': _two_orders('DECIMAL(9, 0)', NUMBER, '999999999'),
        'predict': True,
    },
    'materialize: a sum past 38 digits': {
        'setup': _two_orders('DECIMAL(38, 0)', '9' * 38, NUMBER + '0' * 30),
        'predict': True,
    },
    'materialize: a static decimal target': {
        'setup': _users_column(
            'tier',
            'DECIMAL(18, 2)',
            [f'{NUMBER}.01', f'{NUMBER}.02', f'{NUMBER}.03'],
        ),
        'query': 'PREDICT users.tier FOR EACH users.user_id',
        'anchor_time': None,
        'predict': True,
    },
    'materialize: a decimal average': {
        'setup': _two_orders('DECIMAL(18, 2)', f'{NUMBER}.25', '1.50'),
        'query': 'PREDICT AVG(orders.price, 0, 30, days) FOR EACH users.user_id',
        'predict': True,
    },
    'materialize: a binary question over a sum past its precision': {
        'setup': _two_orders('DECIMAL(9, 0)', NUMBER, '999999999'),
        'query': (
            'PREDICT SUM(orders.price, 0, 30, days) > 5 FOR EACH users.user_id'
        ),
        'predict': True,
    },
    'materialize: a decimal feature float64 cannot hold': {
        'setup': _users_column(
            'f', 'DECIMAL(38, 10)', [f'{NUMBER}123456789.0123456789', '1']
        ),
        'predict': True,
    },
    'materialize: a DECIMAL(38, 38) feature': {
        'setup': _users_column(
            'f', 'DECIMAL(38, 38)', [f'0.{NUMBER}', f'-0.{NUMBER}1']
        ),
        'predict': True,
    },
    'materialize: a decimal key asked for an entity it does not hold': {
        'setup': [
            'ALTER TABLE users ALTER user_id TYPE DECIMAL(18, 0)',
            'ALTER TABLE orders ALTER user_id TYPE DECIMAL(18, 0)',
        ],
        'indices': [0, int(NUMBER)],
    },
    'describe: a decimal declared the time column': {
        'setup': [
            'CREATE OR REPLACE TABLE orders AS SELECT *, '
            f"('{NUMBER}'::DECIMAL(18, 0) + order_id) AS f FROM orders"
        ],
        'time_columns': {'orders': 'f'},
    },
}


def _connection(setup: list[str]) -> Any:
    connection = duckdb.connect()
    for statement in C._shop() + PLANTED + setup:
        connection.execute(statement)
    return connection


def _call(scenario: dict[str, Any]) -> Callable[[], Any]:
    def call() -> Any:
        keywords: dict[str, Any] = {}
        if 'time_columns' in scenario:
            keywords['time_columns'] = scenario['time_columns']
        schema = Schema.from_source(
            DuckDBSource(_connection(scenario.get('setup', []))), **keywords
        )
        anchor = scenario.get('anchor_time', '2024-03-01')
        options: dict[str, Any] = {
            'indices': list(scenario.get('indices', [0, 1]))
        }
        if anchor is not None:
            options['anchor_time'] = pd.Timestamp(anchor)
        request = PredictiveQuery(
            scenario.get('query', QUERY), schema
        ).materialize(**options)
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
        if error is not None:
            # The core's error, never one the bridge makes in its place.
            assert any(where == 'core error' for where, _ in seen.texts), (
                f'{name}: refused outside the core ({error.code})'
            )

    check()


def test_each_sentinel_is_on_the_path() -> None:
    # The data the paths read holds both sentinels: the source hands them
    # over, and a request of the family sends them (a DECIMAL(38, 0)
    # feature whose digits hold the number, which the core sends as the
    # reference does, beside the planted countries).
    setup = _users_column('f', 'DECIMAL(38, 0)', [NUMBER + '0' * 30, '1'])
    connection = _connection(setup)
    countries = {
        c for (c,) in connection.execute('SELECT country FROM users').fetchall()
    }
    features = {
        str(f) for (f,) in connection.execute('SELECT f FROM users').fetchall()
    }
    assert DATA in countries and any(NUMBER in f for f in features)
    for scenario in SCENARIOS.values():
        held = ' '.join(scenario.get('setup', [])) + repr(
            scenario.get('indices')
        )
        assert NUMBER in held, scenario
    with C._endpoint().serve() as (url, received), warnings.catch_warnings():
        # The journey's own warnings, which the scenarios watch.
        warnings.simplefilter('ignore')
        schema = Schema.from_source(DuckDBSource(_connection(setup)))
        request = PredictiveQuery(QUERY, schema).materialize(
            indices=[0, 3, 5], anchor_time=pd.Timestamp('2024-03-01')
        )
        with SDMClient(url, max_retries=0) as client:
            client.predict(request, model=KumoRelational())
    [(_, _, body)] = received.requests
    assert DATA.encode() in body
    assert NUMBER.encode() in body


def test_the_watch_catches_each_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(C, 'SENTINELS', (*C.SENTINELS, NUMBER))
    seen = C._Seen()
    seen.add('str', f'the sum {NUMBER}999 is past DECIMAL(9, 0)')
    seen.add('details', f"{{'country': '{DATA}'}}")
    assert {sentinel for _, sentinel in seen.leaks()} == {NUMBER, DATA}
