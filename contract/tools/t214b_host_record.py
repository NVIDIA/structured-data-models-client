#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Records what T2.14b's host facts give on the reference into
``contract/t214b-host-vectors.json``.

Written by T2.14b's test author, before the implementation. Run with the
reference checkout's interpreter (the environment of
``contract/reference-env.txt``, the ``ref-pandas3`` cell), from this
repository's root::

    <reference venv>/bin/python -I contract/tools/t214b_host_record.py \
        --reference <reference checkout>

The facts the core already returns, which T2.14b reads from the core
instead of re-deriving them in Python (the T2.14a ledger's P2-4 and P2-2
rows), each as the public calls show it, over one DuckDB connection holding
the shop base and the ranking tables of ``rank.top-k`` (:data:`SETUP`):

- ``facts``: for each query of :data:`QUERIES`, the materialized request's
  ``entity_table`` and whether its target moves with time (``_windowed``,
  which the reference's ``Request`` holds under that name), a forecast's
  and a ranking's for each part too;
- ``steps``: a forecast of two steps whose call ``fail_call`` the endpoint
  fails, and a ranking of three candidates likewise: the refusal's class,
  code, details and message, which name the step or the candidate;
- ``repeats``: a ranking given candidates that repeat by Python's equality
  (:data:`CANDIDATES`: ``1``, ``1.0``, ``True``, ``Decimal('1')`` and
  ``np.int64(1)`` all one key of ``dict.fromkeys``): the candidate each
  call asks about, in order, and the result's rankings, or the refusal.

:func:`observe` is imported by ``contract/tests/test_t214b_host.py``, so
``python-core`` is observed by the same code, with the core routes forced.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'contract/t214b-host-vectors.json'
FORMAT = 'sdm-t214b-host-vectors/1'

SETUP = (
    'CREATE TABLE users AS SELECT i::BIGINT AS user_id, (20 + i)::INTEGER '
    "AS age, ['US', 'DE', 'IN'][1 + i % 3] AS country FROM range(8) t(i)",
    'CREATE TABLE orders AS SELECT i::BIGINT AS order_id, (i % 8)::BIGINT '
    'AS user_id, (10 + i * 1.5)::DOUBLE AS price, '
    "TIMESTAMP '2023-01-01' + to_days((i * 7)::INTEGER) AS ts "
    'FROM range(80) t(i)',
    "CREATE TABLE items AS SELECT i::BIGINT AS item_id, ['a', 'b', 'c', "
    "'d'][1 + i] AS kind FROM range(4) t(i)",
    'CREATE TABLE buys AS SELECT i::BIGINT AS buy_id, (i % 8)::BIGINT AS '
    'user_id, ((i * 3) % 4)::BIGINT AS item_id, '
    "TIMESTAMP '2023-01-01' + to_days((i * 5)::INTEGER) AS ts "
    'FROM range(120) t(i)',
)
ANCHOR = '2023-06-01'
RANKING = (
    'PREDICT LIST_DISTINCT(buys.item_id, 0, 30, days) RANK TOP 2 '
    'FOR EACH users.user_id'
)
FORECAST = (
    'PREDICT SUM(orders.price, 0, 30, days) FORECAST 2 TIMEFRAMES '
    'FOR EACH users.user_id'
)
#: The queries whose facts are recorded, by name: windowed and not, an
#: entity of another table, a forecast and a ranking.
QUERIES = {
    'windowed': 'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id',
    'static': 'PREDICT users.age FOR EACH users.user_id',
    'condition': (
        'PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR EACH users.user_id'
    ),
    'other-entity': 'PREDICT orders.price FOR EACH orders.order_id',
    'forecast': FORECAST,
    'ranking': RANKING,
}
#: The candidate lists of ``repeats``, by name, as Python values (built
#: here, since JSON holds neither a Decimal nor a NumPy value).
CANDIDATES = {
    'ints': lambda: [2, 1, 2, 1, 0],
    'int-float-bool': lambda: [1, 1.0, True, 2],
    'bool-first': lambda: [True, 1, 2],
    'zero-false': lambda: [0, False, 1],
    'decimal-first': lambda: [_decimal('1'), 1, 2],
    'numpy-first': lambda: [_numpy(1), 1, 2],
    'float-first': lambda: [1.0, 1, 2],
}


def _decimal(text: str) -> Any:
    from decimal import Decimal

    return Decimal(text)


def _numpy(value: int) -> Any:
    import numpy as np

    return np.int64(value)


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def runner() -> ModuleType:
    return _module('t214b_run_python', ROOT / 'contract/runners/run_python.py')


def endpoint() -> ModuleType:
    return _module(
        'scripted_endpoint', ROOT / 'contract/runners/scripted_endpoint.py'
    )


def as_json(value: Any) -> Any:
    r"""``value`` as JSON writes it, a value of no JSON type as its type
    and text, so a ``Decimal`` and a NumPy value stay told apart."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return None if value != value else value
    if isinstance(value, dict):
        return {str(k): as_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_json(v) for v in value]
    return {'type': type(value).__name__, 'text': str(value)}


def _connection() -> Any:
    import duckdb

    connection = duckdb.connect()
    for statement in SETUP:
        connection.execute(statement)
    return connection


def _schema(connection: Any) -> Any:
    from sdm_client import Schema
    from sdm_connectors import DuckDBSource

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return Schema.from_source(DuckDBSource(connection))


def _materialize(schema: Any, pql: str, **options: Any) -> Any:
    import pandas as pd
    from sdm_client import PredictiveQuery

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return PredictiveQuery(pql, schema).materialize(
            indices=[0, 1], anchor_time=pd.Timestamp(ANCHOR), **options
        )


def _refusal(error: BaseException) -> dict[str, Any]:
    details = getattr(error, 'details', None)
    return {
        'error': type(error).__name__,
        'code': getattr(error, 'code', None),
        'details': as_json(details if isinstance(details, dict) else {}),
        'message': str(error),
    }


def _facts(request: Any) -> dict[str, Any]:
    return {
        'entity_table': request.entity_table,
        'windowed': bool(request._windowed),
    }


def facts(schema: Any) -> dict[str, Any]:
    r"""Each query's request facts, and its parts' where it has parts."""
    out = {}
    for name, pql in QUERIES.items():
        options = {'candidates': [0, 1, 2]} if name == 'ranking' else {}
        try:
            request = _materialize(schema, pql, **options)
        except Exception as error:  # noqa: BLE001 - recorded
            out[name] = {'refusal': _refusal(error)}
            continue
        out[name] = {
            **_facts(request),
            'parts': None
            if request.parts is None
            else [_facts(part) for part in request.parts],
        }
    return out


def _predicted(request: Any, **told: Any) -> dict[str, Any]:
    from sdm_client import KumoRelational, SDMClient

    observed: dict[str, Any] = {}
    with (
        endpoint().serve(**told) as (url, received),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter('ignore')
        try:
            with SDMClient(url) as client:
                result = client.predict(request, model=KumoRelational())
            frame = result.to_pandas()
            observed['rankings'] = (
                as_json(list(frame['rankings']))
                if 'rankings' in frame
                else None
            )
        except Exception as error:  # noqa: BLE001 - recorded
            observed['refusal'] = _refusal(error)
        observed['asked'] = [
            json.loads(body)['metadata']['source_query']
            for _, _, body in received.requests
        ]
    return observed


def steps(schema: Any) -> dict[str, Any]:
    r"""A refused call of a forecast and of a ranking, said of its part."""
    out = {}
    for fail in (0, 1):
        out[f'forecast-fail-{fail}'] = _predicted(
            _materialize(schema, FORECAST), fail_call=fail
        )
        out[f'ranking-fail-{fail}'] = _predicted(
            _materialize(schema, RANKING, candidates=[0, 1, 2]),
            fail_call=fail,
            probabilities=True,
        )
    return out


def repeats(schema: Any) -> dict[str, Any]:
    r"""Each candidate list's questions and answer."""
    out = {}
    for name, made in CANDIDATES.items():
        try:
            request = _materialize(schema, RANKING, candidates=made())
        except Exception as error:  # noqa: BLE001 - recorded
            out[name] = {'refusal': _refusal(error), 'asked': []}
            continue
        out[name] = _predicted(request, probabilities=True)
    return out


def observe(core: bool) -> dict[str, Any]:
    r"""Every scenario, on the core where ``core`` (the DuckDB from-source,
    materialize and predict paths forced, as ``run_python.py`` forces a
    case's routes), else on the routes the interpreter's client takes."""
    if core:
        from sdm_client import _routing

        os.environ['SDM_CORE_ROUTES'] = ','.join(
            _routing.FROM_SOURCE_DUCKDB
            + _routing.MATERIALIZE_DUCKDB
            + _routing.PREDICT_DUCKDB
        )
    try:
        connection = _connection()
        try:
            schema = _schema(connection)
            return {
                'facts': facts(schema),
                'steps': steps(schema),
                'repeats': repeats(schema),
            }
        finally:
            connection.close()
    finally:
        os.environ.pop('SDM_CORE_ROUTES', None)


def load() -> dict[str, Any]:
    r"""The vector file, as committed."""
    return json.loads(OUT.read_text(encoding='utf-8'))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--reference', type=Path, required=True)
    arguments = parser.parse_args(argv)
    run = runner()
    guard = run._Guard(arguments.reference)
    guard.check('before the first case')
    run._require_reference_client(arguments.reference)
    commit = run._git(arguments.reference)
    if not commit['clean']:
        print('the reference checkout is not clean', file=sys.stderr)
        return 1
    document = {
        'format': FORMAT,
        'about': (
            "What T2.14b's host facts give on the reference's public calls: "
            "each query's request entity_table and whether its target moves "
            "with time; a forecast's and a ranking's refused call, said of "
            'its part; and a ranking given candidates that repeat by '
            "Python's equality, each question asked and the rankings. "
            'contract/tests/test_t214b_host.py holds python-core to them, '
            'observed by the same code (observe).'
        ),
        'recorded_by': 'contract/tools/t214b_host_record.py',
        'reference': commit['sha'],
        **observe(core=False),
    }
    guard.check('after the last case')
    OUT.write_text(
        json.dumps(document, indent=1, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
