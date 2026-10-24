#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Writes the ``Request`` pickles an older client makes of a request it
gathered over a DuckDB source, and what that client answers when it
predicts each (T3.26, owner decision 2026-10-23: requests saved by older
clients stay predictable on the core).

    python -I contract/fixtures/old-request-pickles/write.py \
        --client reference --checkout CHECKOUT [--write]
    python -I contract/fixtures/old-request-pickles/write.py \
        --client phase-2 --checkout CHECKOUT [--write]

Run by the interpreter of the client named, installed from ``CHECKOUT``:
``reference``, the reference commit ``contract/reference-env.txt`` names,
whose requests are gathered on its only route; ``phase-2``, the Phase 2
merge (:data:`PHASE_2`), whose requests are gathered on the core's route,
every core path forced, so each holds the state that client's core wrote
beside it. Each request of :func:`objects` is pickled with protocol 4 into
``<client>-<name>.pkl``, and ``expected.json`` records, under the client's
name, what that client answers predicting each as it loads back from the
pickle, on its old route, against the scripted endpoint
(:func:`predicted`).

A request gathered over a source holds the source, which no client pickles
(``Source.__reduce__`` refuses an open connection), so each schema in the
request is pickled without its source and without the engine's reading of
it, as ``Schema.__setstate__`` restores a schema pickled before it could be
bound to one (:func:`_without_the_source`); the test binds each to a source
again in a schema of its tables, so the call routes as one over a source
does. ``contract/tests/test_t326_old_requests_predict_on_the_core.py``
loads each pickle in the client under test and holds its answer on the
core to the recorded one.

The script refuses to run but from the checkout given, at the client's
commit. Without ``--write`` it builds the pickles and compares them with
the committed ones, and the exit status says whether they are the same; CI
never runs it.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import pickle
import re
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REFERENCE_ENV = ROOT / 'contract/reference-env.txt'
ENDPOINT = ROOT / 'contract/runners/scripted_endpoint.py'
#: The Phase 2 merge into ``rust-core``, the client a user of Phase 2 ran.
PHASE_2 = '3ae4d1afc9d11b8e2781d6f76072f2f34281e7a0'
CLIENTS = ('reference', 'phase-2')
#: The source every request is gathered over: a shop and its purchases.
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
#: Each request pickled, by name: its query and materialize's arguments.
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
#: What each prediction asks for: the model's own outputs, and outputs
#: the endpoint answers in part (an explanation, of a call of one row) and
#: leaves out in part (quantiles, a notice on both routes).
ASKED: dict[str, tuple[list[str] | None, ...]] = {
    'regression': (None, ['prediction', 'explanation', 'quantiles']),
    'batched': (None, ['prediction', 'quantiles']),
    'binary': (None, ['prediction', 'probabilities']),
    'forecast': (None, ['prediction', 'explanation', 'quantiles']),
    'ranking': (None, ['rankings']),
}


def _commit(client: str) -> str:
    if client == 'phase-2':
        return PHASE_2
    for line in REFERENCE_ENV.read_text(encoding='utf-8').splitlines():
        if line.startswith('reference: '):
            return line.split(': ', 1)[1].strip()
    raise SystemExit('reference-env.txt names no reference commit')


def endpoint() -> Any:
    name = 'scripted_endpoint'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ENDPOINT)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def source() -> Any:
    r"""A DuckDB source holding :data:`SQL`'s tables."""
    import duckdb
    from sdm_connectors import DuckDBSource

    connection = duckdb.connect()
    for statement in SQL:
        connection.execute(statement)
    return DuckDBSource(connection)


def objects() -> dict[str, Any]:
    r"""Each request pickled, by name, gathered on the route the client
    runs it on by default (the reference's only one; Phase 2's core, its
    paths forced by the caller)."""
    import pandas as pd
    from sdm_client import PredictiveQuery, Schema

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_source(source(), TABLES)
        return {
            name: PredictiveQuery(query, schema).materialize(
                anchor_time=pd.Timestamp(ANCHOR), **arguments
            )
            for name, (query, arguments) in REQUESTS.items()
        }


def _without_the_source(buffer: io.BytesIO) -> pickle.Pickler:
    r"""A pickler writing every schema without its source and the engine's
    reading of it, the state a schema pickled before it could hold either
    has."""
    from sdm_client import Schema

    class _WithoutTheSource(pickle.Pickler):
        def reducer_override(self, obj: Any) -> Any:
            if type(obj) is Schema:
                state = {
                    key: value
                    for key, value in vars(obj).items()
                    if key not in ('_source', '_engine_tables')
                }
                return object.__new__, (Schema,), state
            return NotImplemented

    return _WithoutTheSource(buffer, protocol=4)


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--client', choices=CLIENTS, required=True)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--write', action='store_true')
    given = parser.parse_args(argv)
    checkout = given.checkout.resolve()
    head = subprocess.run(
        ['git', '-C', str(checkout), 'rev-parse', 'HEAD'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if head != _commit(given.client):
        raise SystemExit(f'{checkout} is at {head}, not {given.client}')
    import pandas as pd
    import sdm_client

    client = Path(sdm_client.__file__).resolve()
    if not client.is_relative_to(checkout):
        raise SystemExit(f'sdm_client is {client}, not the checkout given')
    if given.client == 'phase-2':
        # Phase 2 gathers on the core with every core path forced; the
        # reference has one route, and no routing seam.
        from sdm_client import _routing

        os.environ[_routing.ENVIRONMENT] = ','.join(
            sorted(_routing.CORE_ROUTED)
        )
        made = objects()
        del os.environ[_routing.ENVIRONMENT]
    else:
        made = objects()
    path = HERE / 'expected.json'
    expected: dict[str, Any] = (
        json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    )
    recorded: dict[str, Any] = {
        'commit': head,
        'python': '.'.join(map(str, sys.version_info[:3])),
        'pandas': pd.__version__,
        'protocol': 4,
        'requests': {},
    }
    written: dict[str, bytes] = {}
    for name, request in made.items():
        buffer = io.BytesIO()
        _without_the_source(buffer).dump(request)
        written[f'{given.client}-{name}.pkl'] = buffer.getvalue()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            loaded = pickle.loads(buffer.getvalue())
        recorded['requests'][name] = answers(loaded, name)
    expected[given.client] = recorded
    written['expected.json'] = (
        json.dumps(expected, indent=2, sort_keys=True) + '\n'
    ).encode()
    if given.write:
        for file, data in written.items():
            (HERE / file).write_bytes(data)
        return 0
    same = all(
        (HERE / file).exists() and (HERE / file).read_bytes() == data
        for file, data in written.items()
    )
    print("the committed pickles are the client's" if same else 'differ')
    return 0 if same else 1


if __name__ == '__main__':
    sys.exit(main())
