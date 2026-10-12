# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Record the reference's payload rules for ``test_t128a_payload_rules.py``.

Run with the reference checkout's interpreter (the reference environment of
``contract/reference-env.txt``), from this repository's root::

    ~/code/sdmc-reference/.venv/bin/python \\
        packages/structured-data-models-client/tests/t128a_payload_rules_record.py

Each case is a DuckDB source (``contract/bases.yaml``'s ``shop``, changed by
the case's statements, or one generated from a seed here), a query and the
arguments of ``PredictiveQuery.materialize``. The reference materializes it,
and this script keeps what T1.28a's exits compare (``observe``): the task
(its kind, classes, their order and the positive class), the inference and
output sections, the metadata's counts, the context's and the predicted
rows, the instance ids each related table keeps (what the context drop
takes out), the schema section, and the notices the call gave; or, when
the reference refuses, the refusal's class and code. A case whose ``call``
is set is also predicted, against a local endpoint that records the body
the client sends and refuses it, and the body's ``model``, ``output`` and
``inference`` are kept. Nothing here is written by hand but the inputs;
every expected value is the reference's.

``observe`` is imported by the test, so both routes are observed by the
same code.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
import threading
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).with_name('t128a-payload-rules.json')

BINARY = 'PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR EACH users.user_id'
MULTICLASS = 'PREDICT users.country FOR EACH users.user_id'
REGRESSION = 'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id'
ANCHOR = '2024-03-01'

# The exits, as the brief words them; each case names the one it covers.
CLASSIFICATION = (
    'classification payload: classes, their order, positive class and the '
    'context-leakage payload fields equal the reference'
)
REGRESSION_FIELDS = (
    "regression payload's parsed fields (statistic, run mode, entity count) "
    'equal the reference'
)
LIMITS = (
    "payload limits and num_context_samples' bound: the reference's "
    'refusals at the limit'
)
REFUSALS = "empty indices and reserved-name refusals as the reference's"
INFERENCE = 'inference_config written into the call as the reference does'
DECIMAL = "a DECIMAL(38,0) feature's wire dtype as the reference sends it"
AVG_NULL = 'AVG(...) = NULL is a named refusal (the reference crashes)'


def shop() -> list[str]:
    r"""``contract/bases.yaml``'s ``shop`` statements."""
    lines = (
        (ROOT / 'contract/bases.yaml').read_text(encoding='utf-8').splitlines()
    )
    start = lines.index('  shop:') + 1
    statements = []
    for line in lines[start:]:
        if not line.startswith('  - '):
            break
        statements.append(line[len('  - ') :])
    return statements


def generated(seed: int, users: int = 0) -> list[str]:
    r"""A seeded source: ``users`` (a key, an age, a VARCHAR segment of 3 to
    6 classes, a BOOLEAN ``active``) and ``orders`` (a price, a timestamp)
    over 2023 and early 2024."""
    r = random.Random(seed)
    n = users or r.randint(10, 40)
    classes = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta'][
        : r.randint(3, 6)
    ]
    rows = []
    for i in range(n):
        rows.append(
            f"({i}, {r.randint(18, 80)}, '{r.choice(classes)}', "
            f'{"TRUE" if r.random() < 0.5 else "FALSE"})'
        )
    orders = []
    for i in range(r.randint(n * 2, n * 8)):
        day = r.randint(0, 420)
        orders.append(
            f'({i}, {r.randrange(n)}, {round(r.uniform(1, 200), 2)}, '
            f"TIMESTAMP '2023-01-01' + to_days({day}))"
        )
    return [
        'CREATE TABLE users (user_id BIGINT PRIMARY KEY, age INTEGER, '
        'segment VARCHAR, active BOOLEAN)',
        f'INSERT INTO users VALUES {", ".join(rows)}',
        'CREATE TABLE orders (order_id BIGINT PRIMARY KEY, user_id BIGINT '
        'REFERENCES users(user_id), price DOUBLE, ts TIMESTAMP)',
        f'INSERT INTO orders VALUES {", ".join(orders)}',
    ]


def many(users: int) -> list[str]:
    r"""``shop``'s shape with ``users`` users and 900 orders: fewer rows
    than the core's preview reads, so the time column is inferred alike."""
    return [
        f'CREATE TABLE users AS SELECT i::BIGINT AS user_id, (20 + i % 50)'
        f'::INTEGER AS age FROM range({users}) t(i)',
        f'CREATE TABLE orders AS SELECT i::BIGINT AS order_id, (i % {users})'
        f'::BIGINT AS user_id, (10 + (i % 97) * 1.5)::DOUBLE AS price, '
        f"TIMESTAMP '2023-06-01' + to_days((i % 270)::INTEGER) AS ts "
        f'FROM range(900) t(i)',
    ]


def star(dimensions: int) -> list[str]:
    r"""``users`` and ``dimensions`` event tables each referring to it: one
    hop carries ``dimensions + 1`` tables."""
    out = [
        'CREATE TABLE users AS SELECT i::BIGINT AS user_id, (20 + i)::INTEGER '
        'AS age FROM range(8) t(i)',
        'CREATE TABLE orders AS SELECT i::BIGINT AS order_id, (i % 8)::BIGINT '
        'AS user_id, (10 + i * 1.5)::DOUBLE AS price, TIMESTAMP '
        "'2023-01-01' + to_days((i * 7)::INTEGER) AS ts FROM range(80) t(i)",
    ]
    for d in range(1, dimensions):
        out += [
            f'CREATE TABLE events_{d:02d} (event_id BIGINT PRIMARY KEY, '
            f'user_id BIGINT, v_{d} DOUBLE, ts TIMESTAMP)',
            f'INSERT INTO events_{d:02d} SELECT i, i % 8, i * {d}, '
            f"TIMESTAMP '2023-01-01' + to_days((i * 5)::INTEGER) "
            f'FROM range(40) t(i)',
        ]
    return out


CASES: list[dict[str, Any]] = []


def case(
    name: str,
    covers: str,
    query: str,
    setup: list[str],
    **materialize: Any,
) -> None:
    call = materialize.pop('call', False)
    measure = materialize.pop('measure', False)
    CASES.append(
        {
            'measure': measure,
            'name': name,
            'covers': covers,
            'query': query,
            'setup': setup,
            'materialize': materialize,
            'call': call,
        }
    )


# --- the classification payload and its context leakage ------------------
case(
    'base.binary',
    CLASSIFICATION,
    BINARY,
    shop(),
    indices=[0, 1],
    anchor_time=ANCHOR,
)
case('base.multiclass', CLASSIFICATION, MULTICLASS, shop(), indices=[0, 1])
for seed in (11, 12, 13):
    case(
        f'generated-{seed}.binary-count',
        CLASSIFICATION,
        'PREDICT COUNT(orders.*, 0, 30, days) > 2 FOR EACH users.user_id',
        generated(seed),
        indices=[0, 1, 2],
        anchor_time=ANCHOR,
        random_seed=seed,
    )
    case(
        f'generated-{seed}.binary-sum',
        CLASSIFICATION,
        'PREDICT SUM(orders.price, 0, 60, days) > 150 FOR EACH users.user_id',
        generated(seed),
        indices=[3, 1],
        anchor_time=ANCHOR,
        random_seed=seed,
    )
    case(
        f'generated-{seed}.multiclass-static',
        CLASSIFICATION,
        'PREDICT users.segment FOR EACH users.user_id',
        generated(seed),
        indices=[0, 2, 4],
        random_seed=seed,
    )
    case(
        f'generated-{seed}.boolean-static',
        CLASSIFICATION,
        'PREDICT users.active FOR EACH users.user_id',
        generated(seed),
        indices=[1, 0],
        random_seed=seed,
    )
    case(
        f'generated-{seed}.binary-static-condition',
        CLASSIFICATION,
        'PREDICT users.age > 40 FOR EACH users.user_id',
        generated(seed),
        indices=[5, 6],
        random_seed=seed,
    )

# --- the regression's statistic, run mode and entity count ---------------
case(
    'base.regression',
    REGRESSION_FIELDS,
    REGRESSION,
    shop(),
    indices=[0, 1],
    anchor_time=ANCHOR,
)
for statistic in ('mean', 'median'):
    case(
        f'base.regression.statistic-{statistic}',
        REGRESSION_FIELDS,
        REGRESSION,
        shop(),
        indices=[0, 1],
        anchor_time=ANCHOR,
        prediction_statistic=statistic,
    )
case(
    'base.regression.statistic-unknown',
    REGRESSION_FIELDS,
    REGRESSION,
    shop(),
    indices=[0, 1],
    anchor_time=ANCHOR,
    prediction_statistic='mode',
)
case(
    'base.binary.statistic',
    REGRESSION_FIELDS,
    BINARY,
    shop(),
    indices=[0, 1],
    anchor_time=ANCHOR,
    prediction_statistic='median',
)
for samples in (1, 5, 1000, 1001, 10000):
    case(
        f'base.regression.num-context-{samples}',
        REGRESSION_FIELDS,
        REGRESSION,
        shop(),
        indices=[0, 1],
        anchor_time=ANCHOR,
        num_context_samples=samples,
    )
case(
    'many-1001.entities-1000',
    REGRESSION_FIELDS,
    REGRESSION,
    many(1001),
    indices=list(range(1000)),
    anchor_time=ANCHOR,
)

# --- the limits ------------------------------------------------------------
for samples in (0, 10001):
    case(
        f'base.regression.num-context-{samples}',
        LIMITS,
        REGRESSION,
        shop(),
        indices=[0, 1],
        anchor_time=ANCHOR,
        num_context_samples=samples,
    )
case(
    'base.regression.entities-1001',
    LIMITS,
    REGRESSION,
    shop(),
    indices=list(range(1001)),
    anchor_time=ANCHOR,
)
case(
    'many-1001.entities-1001',
    LIMITS,
    REGRESSION,
    many(1001),
    indices=list(range(1001)),
    anchor_time=ANCHOR,
)
case(
    'star.tables-15',
    LIMITS,
    REGRESSION,
    star(14),
    indices=[0, 1],
    anchor_time=ANCHOR,
)
case(
    'star.tables-16',
    LIMITS,
    REGRESSION,
    star(15),
    indices=[0, 1],
    anchor_time=ANCHOR,
)


def padded(length: int) -> list[str]:
    r"""``shop`` with each user carrying ``length`` ASCII characters of
    text: the payload grows by a fixed count per character (each user row
    the context and the predicted rows carry), with nothing to escape, so
    the bytes sent and the reference's ``json.dumps`` count agree (PD-2)."""
    return [
        *shop(),
        f"CREATE OR REPLACE TABLE users AS SELECT *, repeat('a', {length}) "
        'AS note FROM users',
    ]


def at_the_size_limit() -> int:
    r"""The longest text whose payload the reference takes: the payload
    of ``padded(n)`` is linear in ``n``, solved from two sizes and checked
    on both sides of the limit by ``main``."""
    from sdm_relational_engine.rfm.inference.payload import (
        payload_size_bytes,
    )

    from sdm_client import PredictiveQuery, Schema

    def size(length: int) -> int:
        import duckdb
        import pandas as pd
        from sdm_connectors import DuckDBSource

        connection = duckdb.connect()
        for statement in padded(length):
            connection.execute(statement)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            schema = Schema.from_source(DuckDBSource(connection))
            request = PredictiveQuery(REGRESSION, schema).materialize(
                indices=[0, 1], anchor_time=pd.Timestamp(ANCHOR)
            )
        return payload_size_bytes(request._payloads[0])

    base, step = size(0), size(1000) - size(0)
    assert step % 1000 == 0, step
    return (PAYLOAD_LIMIT - base) // (step // 1000)


#: The engine's payload limit (``rfm.py``'s ``_MAX_SIZE``).
PAYLOAD_LIMIT = 30 * 1024 * 1024


# --- the empty and reserved-name refusals ----------------------------------
case(
    'empty.indices',
    REFUSALS,
    REGRESSION,
    shop(),
    indices=[],
    anchor_time=ANCHOR,
)
case(
    'reserved.instance-id',
    REFUSALS,
    REGRESSION,
    [*shop(), 'ALTER TABLE users ADD COLUMN instance_id BIGINT'],
    indices=[0, 1],
    anchor_time=ANCHOR,
)
case(
    'reserved.anchor-time',
    REFUSALS,
    REGRESSION,
    [*shop(), 'ALTER TABLE orders ADD COLUMN __kumo_anchor_time DOUBLE'],
    indices=[0, 1],
    anchor_time=ANCHOR,
)

# --- inference_config, as the AdventureWorks notebook passes it -------------
case(
    'base.regression.inference-config',
    INFERENCE,
    REGRESSION,
    shop(),
    indices=[0, 1],
    anchor_time=ANCHOR,
    inference_config={'num_estimators': 4, 'column_shuffle': True},
    call=True,
)
case(
    'base.binary.inference-config',
    INFERENCE,
    BINARY,
    shop(),
    indices=[0, 1],
    anchor_time=ANCHOR,
    inference_config={'num_estimators': 4, 'column_shuffle': True},
    call=True,
)
case(
    'base.regression.inference-config-unread',
    INFERENCE,
    REGRESSION,
    shop(),
    indices=[0, 1],
    anchor_time=ANCHOR,
    inference_config={'num_estimators': 4, 'output_type': 'median'},
    call=True,
)

# --- a DECIMAL(38, 0) feature (decimal.feature-38-0-wide) -------------------
case(
    'decimal.feature-38-0-wide',
    DECIMAL,
    REGRESSION,
    [
        *shop(),
        'CREATE OR REPLACE TABLE users AS SELECT *, '
        '(12345678901234567890123456789::DECIMAL(38, 0) + user_id) AS f '
        'FROM users',
    ],
    indices=[0, 1],
    anchor_time=ANCHOR,
)

# --- AVG(...) = NULL ---------------------------------------------------------
case(
    'avg-equals-null',
    AVG_NULL,
    'PREDICT AVG(orders.price, 0, 30, days) = NULL FOR EACH users.user_id',
    shop(),
    indices=[0, 1],
    anchor_time=ANCHOR,
)


def _rows(table: Any) -> Any:
    if not isinstance(table, dict):
        return None
    return {
        'columns': table.get('columns'),
        'rows': json.loads(json.dumps(table.get('rows'), default=str)),
    }


def _instances(related: Any) -> Any:
    r"""Each related table's columns and the instance ids its rows hang
    off, sorted: what the context drop keeps of it."""
    if not isinstance(related, dict):
        return None
    out = {}
    for name, table in sorted(related.items()):
        columns = table.get('columns') or []
        at = columns.index('instance_id') if 'instance_id' in columns else None
        out[name] = {
            'columns': columns,
            'instance_ids': sorted(
                {row[at] for row in table.get('rows') or []}
                if at is not None
                else []
            ),
        }
    return out


@contextmanager
def _refusing_endpoint() -> Iterator[tuple[str, list[bytes]]]:
    r"""An endpoint that records each body and refuses it with a 400."""
    received: list[bytes] = []
    body = json.dumps({'code': 'RECORDED', 'detail': 'recorded'}).encode()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *args: object) -> None:
            pass

        def do_POST(self) -> None:  # noqa: N802 - http.server's name
            length = int(self.headers.get('Content-Length') or 0)
            received.append(self.rfile.read(length))
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_address[1]}', received
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _refusal(error: BaseException) -> dict[str, Any]:
    return {
        'outcome': 'refused',
        'class': type(error).__name__,
        'code': getattr(error, 'code', None),
        # The classes it is an instance of: the core's host may raise a
        # subclass of the reference's class for the same code.
        'mro': [c.__name__ for c in type(error).__mro__],
    }


def observe(
    spec: dict[str, Any], *, predict_too: bool = True
) -> dict[str, Any]:
    r"""The case's outcome on the route the environment selects."""
    import duckdb
    import pandas as pd
    from sdm_connectors import DuckDBSource

    from sdm_client import KumoRelational, PredictiveQuery, Schema, SDMClient

    connection = duckdb.connect()
    for statement in spec['setup']:
        connection.execute(statement)
    arguments = dict(spec['materialize'])
    for key in ('anchor_time', 'context_anchor_time'):
        if key in arguments:
            arguments[key] = pd.Timestamp(arguments[key])
    notices: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            schema = Schema.from_source(DuckDBSource(connection))
            request = PredictiveQuery(spec['query'], schema).materialize(
                **arguments
            )
        notices = sorted(
            str(w.message)
            for w in caught
            if type(w.message).__name__ == 'SDMWarning'
        )
    except Exception as error:  # noqa: BLE001 - the outcome is compared
        out = _refusal(error)
        out['message'] = str(error)
        out['rule'] = (getattr(error, 'details', None) or {}).get('rule')
        return out
    payload = request._payloads[0]
    out: dict[str, Any] = {
        'outcome': 'payload',
        'parts': len(request._payloads),
        'task': payload.get('task'),
        'output': payload.get('output'),
        'inference': payload.get('inference'),
        'metadata': {
            k: v
            for k, v in (payload.get('metadata') or {}).items()
            if k != 'request_id'
        },
        'schema': payload.get('schema'),
        'context': _rows((payload.get('context') or {}).get('instance_table')),
        'context_related': _instances(
            (payload.get('context') or {}).get('related_tables')
        ),
        'predict': _rows((payload.get('predict') or {}).get('instance_table')),
        'predict_related': _instances(
            (payload.get('predict') or {}).get('related_tables')
        ),
        'notices': notices,
    }
    if spec['covers'] == DECIMAL:
        # The DECIMAL(38, 0) column's values as the context sends them.
        users = ((payload.get('context') or {}).get('related_tables') or {})[
            'users'
        ]
        at = users['columns'].index('f')
        out['decimal_values'] = sorted(
            {json.dumps(row[at]) for row in users['rows']}
        )
    if spec['covers'] != CLASSIFICATION:
        # Outside the classification exit only the counts are compared.
        for key in ('context', 'predict'):
            table = out[key] or {}
            out[key] = len(table.get('rows') or [])
        for key in ('context_related', 'predict_related'):
            out[key] = sorted(out[key] or {})
    if spec.get('measure'):
        # The reference's count (``payload_size_bytes``): compact JSON,
        # ASCII here, so it is also the bytes sent.
        out['bytes'] = len(
            json.dumps(payload, allow_nan=False, separators=(',', ':')).encode(
                'utf-8'
            )
        )
        out['schema'] = None
    if spec.get('call') and predict_too:
        with _refusing_endpoint() as (url, received):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    with SDMClient(url, max_retries=0) as client:
                        client.predict(request, model=KumoRelational())
            except Exception as error:  # noqa: BLE001 - recorded
                out['call_refusal'] = _refusal(error)
        if received:
            body = json.loads(received[0])
            out['call'] = {
                key: body.get(key) for key in ('model', 'output', 'inference')
            }
        else:
            out['call'] = None
    return out


def main() -> int:
    import sdm_relational_engine  # noqa: F401 - the reference's engine

    reference = Path.home() / 'code/sdmc-reference'
    commit = subprocess.run(
        ['git', '-C', str(reference), 'rev-parse', 'HEAD'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    longest = at_the_size_limit()
    case(
        'size.at-the-limit',
        LIMITS + ' (PD-2: the bytes counted)',
        REGRESSION,
        padded(longest),
        indices=[0, 1],
        anchor_time=ANCHOR,
        measure=True,
    )
    case(
        'size.one-past-the-limit',
        LIMITS + ' (PD-2: the bytes counted)',
        REGRESSION,
        padded(longest + 1),
        indices=[0, 1],
        anchor_time=ANCHOR,
        measure=True,
    )
    recorded = []
    for spec in CASES:
        expected = observe(spec)
        expected.pop('message', None)
        expected.pop('rule', None)
        recorded.append({**spec, 'expected': expected})
        print(
            spec['name'],
            expected['outcome'],
            expected.get('code'),
            file=sys.stderr,
        )
    OUT.write_text(
        json.dumps(
            {
                'recorded_by': Path(__file__).name,
                'reference': commit,
                'cases': recorded,
            },
            indent=1,
        )
        + '\n',
        encoding='utf-8',
    )
    print(f'{len(recorded)} cases written to {OUT}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
