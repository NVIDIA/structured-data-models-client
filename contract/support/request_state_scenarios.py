#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request state scenarios: independent inputs and current-client test helpers."""

from __future__ import annotations

import importlib.util
import json
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
    'negative-zero': lambda: [0.0, -0.0, 1],
    'decimal-scales': lambda: [_decimal('1.0'), _decimal('1.00'), 2],
    'missing-twice': lambda: [_numpy_nan(), _numpy_nan(), 1],
}


def _decimal(text: str) -> Any:
    from decimal import Decimal

    return Decimal(text)


def _numpy(value: int) -> Any:
    import numpy as np

    return np.int64(value)


def _numpy_nan() -> Any:
    r"""NumPy's one NaN object, which ``dict.fromkeys`` holds equal to
    itself by identity."""
    import numpy as np

    return np.nan


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
        'scripted_endpoint', ROOT / 'tests/support/scripted_endpoint.py'
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


def observe() -> dict[str, Any]:
    """Run each scenario through the installed client."""
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


def load() -> dict[str, Any]:
    r"""The vector file, as committed."""
    return json.loads(OUT.read_text(encoding='utf-8'))
