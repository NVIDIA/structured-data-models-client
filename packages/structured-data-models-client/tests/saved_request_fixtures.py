# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scripted model responses and observations for saved-request compatibility."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
ENDPOINT = ROOT / 'tests/support/scripted_endpoint.py'


ASKED: dict[str, tuple[list[str] | None, ...]] = {
    'regression': (None, ['prediction', 'explanation', 'quantiles']),
    'batched': (None, ['prediction', 'quantiles']),
    'binary': (None, ['prediction', 'probabilities']),
    'forecast': (None, ['prediction', 'explanation', 'quantiles']),
    'ranking': (None, ['rankings']),
}


def endpoint() -> Any:
    name = 'scripted_endpoint'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ENDPOINT)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _digest(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def _plain(value: Any) -> Any:
    r"""``value`` as JSON holds it, a mapping's keys as text, in order."""
    return json.loads(json.dumps(value, default=str))


def _answer(result: Any) -> dict[str, Any]:
    r"""What a result says: each item, its outputs, its explanation, the
    candidates it left unscored and the endpoint's metadata, but the
    request id, which is drawn per call."""
    items = []
    for item in result.items:
        fields = (
            item.model_dump() if hasattr(item, 'model_dump') else vars(item)
        )
        items.append(_plain({k: v for k, v in fields.items() if v is not None}))
    metadata = dict(result.metadata or {})
    metadata.pop('request_id', None)
    return {
        'items': items,
        'outputs': list(result.outputs or ()),
        'explanation': _plain(result.explanation),
        'unscored': _plain(
            list(getattr(result, 'unscored_candidates', None) or [])
        ),
        'metadata': _plain(metadata),
        'frame': _digest(result.to_pandas().astype(str).values.tolist()),
    }


def predicted(request: Any, outputs: list[str] | None) -> dict[str, Any]:
    r"""What predicting ``request`` for ``outputs`` sends and answers,
    against the scripted endpoint answering probabilities, explanations and
    item metadata: each body by its digest (its request id left out), the
    answer (:func:`_answer`) or the refusal's class and code, and the
    notices given, each without the code a core route puts before it."""
    from sdm_client import KumoRelational, SDMClient

    with (
        endpoint().serve(
            probabilities=True, explanation=True, item_metadata=True
        ) as (url, received),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter('always')
        out: dict[str, Any] = {}
        try:
            with SDMClient(url, max_retries=0) as client:
                # This scripted endpoint implements predictions only. Declare
                # that capability so session probing does not change its
                # recorded prediction-call ordinals or payload inventory.
                client._transport.serves_sessions = False
                result = client.predict(
                    request, model=KumoRelational(), outputs=outputs
                )
            out['answer'] = _answer(result)
        except Exception as error:  # noqa: BLE001 - recorded
            out['refusal'] = [
                type(error).__name__,
                getattr(error, 'code', None),
            ]
        bodies = []
        for _, _, body in received.requests:
            parsed = json.loads(body)
            parsed.get('metadata', {}).pop('request_id', None)
            bodies.append(_digest(parsed))
        out['bodies'] = bodies
    out['notices'] = sorted(
        re.sub(r'^\[[A-Z_]+\] ', '', str(w.message))
        for w in caught
        if type(w.message).__name__ == 'SDMWarning'
    )
    return out


def answers(request: Any, name: str) -> list[dict[str, Any]]:
    r"""``request``'s answer to each of :data:`ASKED`'s outputs."""
    return [predicted(request, outputs) for outputs in ASKED[name]]


SQL = (
    'CREATE TABLE users AS SELECT i::BIGINT AS user_id, (20 + i)::INTEGER '
    "AS age, ['US', 'DE', 'IN'][1 + i % 3] AS country FROM range(8) t(i)",
    'CREATE TABLE orders AS SELECT i::BIGINT AS order_id, (i % 8)::BIGINT '
    'AS user_id, (10 + i * 1.5)::DOUBLE AS price, TIMESTAMP '
    "'2023-01-01' + to_days((i * 7)::INTEGER) AS ts FROM range(80) t(i)",
    'CREATE TABLE items AS SELECT i::BIGINT AS item_id, '
    "['a', 'b', 'c', 'd'][1 + i] AS kind FROM range(4) t(i)",
    'CREATE TABLE buys AS SELECT i::BIGINT AS buy_id, (i % 8)::BIGINT AS '
    'user_id, ((i * 3) % 4)::BIGINT AS item_id, TIMESTAMP '
    "'2023-06-01' + to_days((i * 2)::INTEGER) AS ts FROM range(120) t(i)",
)


TABLES = {
    'users': 'users',
    'orders': 'orders',
    'items': 'items',
    'buys': 'buys',
}


ANCHOR = '2023-12-01'


REQUESTS: dict[str, tuple[str, dict[str, Any]]] = {
    'regression': (
        'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id',
        # One entity: the model explains a call of one row only.
        {'indices': [3]},
    ),
    'batched': (
        'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id',
        {'indices': [0, 1, 2, 3, 4, 5], 'rows_per_call': 2},
    ),
    'binary': (
        'PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR EACH users.user_id',
        {'indices': [0, 1, 2, 3]},
    ),
    'forecast': (
        'PREDICT SUM(orders.price, 0, 30, days) FORECAST 2 TIMEFRAMES '
        'FOR EACH users.user_id',
        {'indices': [3]},
    ),
    'ranking': (
        'PREDICT LIST_DISTINCT(buys.item_id, 0, 30, days) RANK TOP 2 '
        'FOR EACH users.user_id',
        # A repeat, so a candidate's position among those given is not its
        # question's; and an item nobody buys, which no context teaches, so
        # it is left out unscored.
        {'indices': [0, 1], 'candidates': [2, 1, 1, 9, 0]},
    ),
}


def source() -> Any:
    r"""A DuckDB source holding :data:`SQL`'s tables."""
    import duckdb
    from sdm_connectors import DuckDBSource

    connection = duckdb.connect()
    for statement in SQL:
        connection.execute(statement)
    return DuckDBSource(connection)
