#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Records the reference's ``AVG(...) = NULL`` and ``!= NULL`` into
``contract/t136-average-null-vectors.json`` (register P1B-21, T1.36).

Run with the reference checkout's interpreter (the environment of
``contract/reference-env.txt``), from this repository's root::

    ~/code/sdmc-reference/.venv/bin/python -I contract/tools/average_null_record.py

Written by T1.36's independent test author before the fix. Each case is a
DuckDB source (``users`` and ``orders`` as ``contract/bases.yaml``'s
``shop`` lays them out, changed by the case's statements), a query whose
target compares a windowed ``AVG`` with ``NULL``, and the arguments of
``PredictiveQuery.materialize``. Every case is asked with both operators.
The reference materializes it and this script keeps what
``contract/tests/test_t136_average_null.py`` compares (``observe``): the
task, the context's and the predicted instance rows (the labels among
them), the instance ids each related table keeps and the metadata's
counts; or, when the reference fails, the failure's class and code and
the engine's exception under it. Nothing here is written by hand but the
inputs; every expected value is the reference's.

What the reference does (``sdm_relational_engine``'s
``PQueryPandasExecutor``): an aggregation across the entity's link is a
``Join``, whose ``execute_join`` asks the aggregation with ``filter_na``
set, so ``AVG`` gives one value per training-trial row whose window holds
a non-null value and none for the others; ``execute_condition`` then, the
constant being ``NULL``, keeps every value it got; ``= NULL`` labels each
``False`` and ``!= NULL`` each ``True``. ``execute`` then indexes the entity
mask (one entry per row) with the condition's (one per value): where the
two counts differ it raises ``IndexError`` (the client's
``INTERNAL_ERROR``), and where there is no value at all, ``ValueError``
from the broadcast after it (the client's ``INVALID_REQUEST``, numpy's
words). So the reference labels a source only when every row of its
training trials has a non-null value in its window. Which rows those are:
the duckdb sampler's ``_sample_target_set`` labels the training trials in
batches of 10,000 rows and stops once it has the run mode's examples
(1,000, or 10,000 past a context of 1,000), so the first batch decides:
the first 10,000 rows of the trial list (context anchor ``anchor_time``
minus the target's timeframe, stepping back by it while the trials are
short of ten times the run mode's examples and the anchor is not before
the aggregated table's earliest time; each anchor's entities those valid
at it, an entity table's time column at or before it). The context size
asked narrows the context after labelling, not the trials; an entity
filter masks rows after labelling, so a row it filters out still has to
have a value; the predicted rows are not labelled, so an empty window
after ``anchor_time`` is no failure.
"""

from __future__ import annotations

import json
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'contract/t136-average-null-vectors.json'
ANCHOR = '2024-03-01'
OPERATORS = ('= NULL', '!= NULL')


#: ``shop``'s users.
def users(count: int = 8) -> str:
    r"""``shop``'s users, ``count`` of them."""
    return (
        'CREATE TABLE users AS SELECT i::BIGINT AS user_id, (20 + i)::INTEGER '
        "AS age, ['US', 'DE', 'IN'][1 + i % 3] AS country "
        f'FROM range({count}) t(i)'
    )


def orders(days: int = 600, every: int = 1, count: int = 8) -> str:
    r"""``shop``'s orders, dense: each of the ``count`` users orders once
    every ``every`` days from 2023-01-01, for ``days`` steps."""
    return (
        'CREATE TABLE orders AS SELECT i::BIGINT AS order_id, '
        f'(i % {count})::BIGINT AS user_id, (10 + i * 1.5)::DOUBLE AS price, '
        f"TIMESTAMP '2023-01-01' + to_days((i // {count} * "
        f'{every})::INTEGER) AS ts FROM range({count * days}) t(i)'
    )


def dense(
    *setup: str, days: int = 600, every: int = 1, count: int = 8
) -> list[str]:
    return [users(count), orders(days, every, count), *setup]


def shop() -> list[str]:
    r"""``contract/bases.yaml``'s ``shop`` statements: an order per user
    every 56 days, so many 30-day windows are empty."""
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


def gap(user: int, after: str, through: str) -> str:
    r"""``user``'s orders in ``(after, through]`` deleted."""
    return (
        f'DELETE FROM orders WHERE user_id = {user} '
        f"AND ts > TIMESTAMP '{after}' AND ts <= TIMESTAMP '{through}'"
    )


def query(window: str = '0, 30, days', where: str = '') -> str:
    return (
        f'PREDICT AVG(orders.price, {window}) {{op}} '
        f'FOR EACH users.user_id{where}'
    )


#: The cases, each asked with both operators. ``labels``: every row of the
#: reference's training trials has a non-null price in its window, so it
#: labels them; ``fails``: some row has none, so it fails.
CASES: list[dict[str, Any]] = []


def case(
    name: str,
    expect: str,
    covers: str,
    pql: str,
    setup: list[str],
    **materialize: Any,
) -> None:
    materialize.setdefault('indices', [0, 1])
    materialize.setdefault('anchor_time', ANCHOR)
    for op in OPERATORS:
        CASES.append(
            {
                'name': f'{name} {op}',
                'expect': expect,
                'covers': covers,
                'query': pql.format(op=op),
                'setup': setup,
                'materialize': materialize,
            }
        )


# --- labelled: every training trial's window holds a non-null price -------
case(
    'dense',
    'labels',
    'an order per user per day: every window holds a price',
    query(),
    dense(),
)
case(
    'dense.context-16',
    'labels',
    'the context size narrows the context after labelling',
    query(),
    dense(),
    num_context_samples=16,
)
case(
    'dense.indices-3-5-7',
    'labels',
    'other entities predicted',
    query(),
    dense(),
    indices=[3, 5, 7],
)
case(
    'dense.half-null',
    'labels',
    'every other day the price is NULL: each window still holds one',
    query(),
    dense('UPDATE orders SET price = NULL WHERE (order_id // 8) % 2 = 0'),
)
case(
    'dense.forward-window',
    'labels',
    'a window that starts after the anchor (10 to 40 days)',
    query('10, 40, days'),
    dense(),
)
case(
    'dense.month',
    'labels',
    'a window in months',
    query('0, 1, months'),
    dense(),
)
case(
    'dense.hours',
    'labels',
    'a window in hours (24): the next day midnight is inside it',
    query('0, 24, hours'),
    dense(),
)
case(
    'weekly.7-days',
    'labels',
    'an order per user per week and a 7-day window: never empty',
    query('0, 7, days'),
    dense(days=100, every=7),
)
case(
    'dense.ends-before-the-window',
    'labels',
    "the data ends 2024-02-29: the predicted rows' window is empty, and "
    'predicted rows are not labelled',
    query(),
    dense(days=425),
)
case(
    'dense.gap-in-the-predicted-window',
    'labels',
    'user 3 orders nothing in (2024-03-01, 2024-03-31], only the predicted '
    "rows' window",
    query(),
    dense(gap(3, '2024-03-01', '2024-03-31')),
    indices=[3, 4],
)
case(
    'dense.gap-before-the-earliest-window',
    'labels',
    'user 3 orders nothing until 2023-01-06, before the earliest training '
    'window (2023-01-06, 2023-02-05]',
    query(),
    dense(gap(3, '2022-12-01', '2023-01-06')),
)
case(
    'dense.user-joins-late.signup',
    'labels',
    "user 7's orders start 2023-12-01, users' time column says it joined "
    'then: the older anchors leave it out',
    query(),
    dense(
        "DELETE FROM orders WHERE user_id = 7 AND ts < TIMESTAMP '2023-12-01'",
        'CREATE OR REPLACE TABLE users AS SELECT *, CASE WHEN user_id = 7 '
        "THEN TIMESTAMP '2023-12-01' ELSE TIMESTAMP '2022-01-01' END "
        'AS signup FROM users',
    ),
)
case(
    'dense.gap-of-29-days',
    'labels',
    'user 3 orders nothing for 29 days, shorter than any window',
    query(),
    dense(gap(3, '2023-06-01', '2023-06-30')),
)

case(
    'hours.30-users.gap-past-the-first-10000-trials',
    'labels',
    "30 users and a 24-hour window: the trials' anchors step back a day "
    'at a time, 30 rows each; user 3 orders nothing on 2023-03-27, the '
    'window of the anchor 340 days back (rows 10,200 to 10,229): past the '
    'first 10,000 trials, which are all the reference labels',
    query('0, 24, hours'),
    dense(gap(3, '2023-03-26', '2023-03-27'), count=30),
)
case(
    'hours.30-users.gap-past-the-first-10000-trials.context-1001',
    'labels',
    'the same with a context of 1,001 (the best run mode, 100,000 trials '
    'asked): the reference still labels the first 10,000 only',
    query('0, 24, hours'),
    dense(gap(3, '2023-03-26', '2023-03-27'), count=30),
    num_context_samples=1_001,
)

# --- failed: some training trial's window holds no non-null price ---------
case(
    'hours.30-users.gap-in-the-first-10000-trials',
    'fails',
    'the same source with the gap on 2023-05-06, the window of the anchor '
    '300 days back (rows 9,000 to 9,029): among the first 10,000 trials',
    query('0, 24, hours'),
    dense(gap(3, '2023-05-05', '2023-05-06'), count=30),
)
case(
    'shop',
    'fails',
    "shop's orders, one per user every 56 days: many 30-day windows empty "
    "(T1.28a's avg-equals-null case)",
    query(),
    shop(),
)
case(
    'dense.gap-in-the-last-window',
    'fails',
    'user 3 orders nothing in (2024-01-31, 2024-03-01], the latest training '
    'window',
    query(),
    dense(gap(3, '2024-01-31', '2024-03-01')),
    indices=[3, 4],
)
case(
    'dense.gap-in-an-older-window',
    'fails',
    'user 3 orders nothing in (2023-08-04, 2023-09-03], a training window '
    'seven anchors back',
    query(),
    dense(gap(3, '2023-08-04', '2023-09-03')),
)
case(
    'dense.gap-in-an-older-window.context-4',
    'fails',
    'the same gap with a context of 4: the trials are not narrowed',
    query(),
    dense(gap(3, '2023-08-04', '2023-09-03')),
    num_context_samples=4,
)
case(
    'dense.gap-of-the-user-not-predicted',
    'fails',
    'user 3, empty in the latest training window, is not among the indices',
    query(),
    dense(gap(3, '2024-01-31', '2024-03-01')),
    indices=[0, 1],
)
case(
    'dense.user-without-orders',
    'fails',
    'a ninth user who never orders',
    query(),
    dense("INSERT INTO users VALUES (8, 28, 'US')"),
)
case(
    'dense.user-joins-late',
    'fails',
    "user 7's orders start 2023-12-01 and users has no time column",
    query(),
    dense(
        "DELETE FROM orders WHERE user_id = 7 AND ts < TIMESTAMP '2023-12-01'"
    ),
)
case(
    'dense.null-prices-in-a-window',
    'fails',
    "user 0's prices are NULL in (2023-06-01, 2023-07-15]: orders but no value",
    query(),
    dense(
        'UPDATE orders SET price = NULL WHERE user_id = 0 '
        "AND ts > TIMESTAMP '2023-06-01' AND ts <= TIMESTAMP '2023-07-15'"
    ),
)
case(
    'weekly.6-days',
    'fails',
    'an order per user per week and a 6-day window: some windows empty',
    query('0, 6, days'),
    dense(days=100, every=7),
)
case(
    'dense.all-null',
    'fails',
    'every price NULL: no row has a value (the reference fails '
    'INVALID_REQUEST, a broadcast error, not INTERNAL_ERROR)',
    query(),
    dense('UPDATE orders SET price = NULL'),
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
    off, sorted."""
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


def observe(spec: dict[str, Any]) -> dict[str, Any]:
    r"""The case's outcome on the route the environment selects."""
    import duckdb
    import pandas as pd
    from sdm_client import PredictiveQuery, Schema
    from sdm_connectors import DuckDBSource

    connection = duckdb.connect()
    for statement in spec['setup']:
        connection.execute(statement)
    arguments = dict(spec['materialize'])
    arguments['anchor_time'] = pd.Timestamp(arguments['anchor_time'])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            schema = Schema.from_source(DuckDBSource(connection))
            request = PredictiveQuery(spec['query'], schema).materialize(
                **arguments
            )
    except Exception as error:  # noqa: BLE001 - the outcome is compared
        cause = error.__cause__
        return {
            'outcome': 'refused',
            'class': type(error).__name__,
            'code': getattr(error, 'code', None),
            'mro': [c.__name__ for c in type(error).__mro__],
            'rule': (getattr(error, 'details', None) or {}).get('rule'),
            'cause': (
                None if cause is None else f'{type(cause).__name__}: {cause}'
            ),
            'message': str(error),
        }
    payload = request._payloads[0]
    metadata = payload.get('metadata') or {}
    return {
        'outcome': 'payload',
        'parts': len(request._payloads),
        'task': payload.get('task'),
        'counts': {
            key: metadata.get(key)
            for key in ('num_context_examples', 'num_prediction_examples')
        },
        'context': _rows((payload.get('context') or {}).get('instance_table')),
        'context_related': _instances(
            (payload.get('context') or {}).get('related_tables')
        ),
        'predict': _rows((payload.get('predict') or {}).get('instance_table')),
        'predict_related': _instances(
            (payload.get('predict') or {}).get('related_tables')
        ),
    }


def _scalar(value: Any) -> bool:
    return not isinstance(value, (dict, list))


def _written(value: Any, depth: int = 0) -> str:
    r"""``value`` as JSON indented one space a level, a list of scalars on
    one line (a row, the instance ids), so the vectors stay small."""
    pad = ' ' * depth
    if isinstance(value, dict) and value:
        items = [
            f'{pad} {json.dumps(key)}: {_written(item, depth + 1)}'
            for key, item in value.items()
        ]
        return '{\n' + ',\n'.join(items) + f'\n{pad}}}'
    if isinstance(value, list) and value and not all(map(_scalar, value)):
        items = [f'{pad} {_written(item, depth + 1)}' for item in value]
        return '[\n' + ',\n'.join(items) + f'\n{pad}]'
    return json.dumps(value)


def main() -> int:
    import sdm_relational_engine  # noqa: F401 - the reference's engine

    reference = Path.home() / 'code/sdmc-reference'
    commit = subprocess.run(
        ['git', '-C', str(reference), 'rev-parse', 'HEAD'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
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
            expected.get('cause'),
            file=sys.stderr,
        )
    OUT.write_text(
        _written(
            {
                'recorded_by': 'contract/tools/average_null_record.py',
                'reference': commit,
                'cases': recorded,
            }
        )
        + '\n',
        encoding='utf-8',
    )
    print(f'{len(recorded)} cases written to {OUT}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
