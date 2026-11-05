# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Label values across regression, classification and exact decimal inputs.

Fixed expected rows retain historical coverage without executing the old client.
Decimal sums, averages and classes use exact expectations where the current core improves precision.
Supported cases must produce labels; historical deferred refusals are not allowed."""

from __future__ import annotations

import json
import warnings
from fractions import Fraction
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pytest
from sdm_connectors import DuckDBSource

from sdm_client import PredictiveQuery, Schema

ROOT = Path(__file__).resolve().parents[3]
RECORDED = json.loads(
    Path(__file__)
    .with_name('labels_reference.json')
    .read_text(encoding='utf-8')
)

#: Register P2B-1 (exact decimals): since main's eb6c68eb the
#: reference sums and averages a DECIMAL column as float64, where the core
#: keeps exact decimals. On these cases the core's labels are held to the
#: mathematically exact aggregate over the case's own rows, rounded once to
#: float32, and everything else to the reference's.
EXACT_DECIMAL = {
    'sum-decimal': 'SUM',
    'avg-decimal': 'AVG',
    # Sums at float32's midpoint 1 + 2**-24: rounded once, user 1's
    # 1.000000059604644776 is the next float32 up, where the reference's
    # float64-then-float32 reading gives 1.0.
    'sum-decimal-midpoint': 'SUM',
}
#: The same register entry for a decimal class target: one class per
#: distinct exact value, where the reference's float reading merges
#: values that differ past float32.
EXACT_CLASSES = {'multiclass-static-decimal': 'tier'}


def _float32_once(exact: Fraction) -> float:
    r"""The float32 nearest ``exact``, ties to even: one rounding, never
    through a float64 first."""
    import numpy as np

    first = np.float32(float(exact))
    candidates = {
        first,
        np.nextafter(first, np.float32(np.inf)),
        np.nextafter(first, np.float32(-np.inf)),
    }

    def key(c: Any) -> tuple[Fraction, int]:
        bits = int(np.array(c, dtype=np.float32).view(np.uint32))
        return (abs(Fraction(float(c)) - exact), bits & 1)

    return float(min(candidates, key=key))


def _exact_labels(spec: dict[str, Any], rows: list[list[Any]]) -> list[float]:
    r"""Each context row's label as the exact SUM or AVG of its entity's
    orders in the window after its anchor (DuckDB's exact decimal SUM and
    COUNT, divided as a fraction), rounded once to float32."""
    aggregate = EXACT_DECIMAL[spec['name']]
    assert spec['query'] == (
        f'PREDICT {aggregate}(orders.amount, 0, 30, days) FOR EACH users.user_id'
    )
    connection = duckdb.connect()
    for statement in spec['setup']:
        connection.execute(statement)
    labels = []
    for _, entity, anchor, _ in rows:
        total, count = connection.execute(
            'SELECT SUM(amount), COUNT(amount) FROM orders WHERE user_id = ? '
            'AND ts > CAST(? AS TIMESTAMP) '
            'AND ts <= CAST(? AS TIMESTAMP) + INTERVAL 30 DAY',
            [entity, anchor[:19], anchor[:19]],
        ).fetchone()
        if aggregate == 'SUM':
            # An empty window sums to zero.
            exact = Fraction(total) if count else Fraction(0)
        else:
            assert count, (entity, anchor, 'an average over no order')
            exact = Fraction(total) / count
        labels.append(_float32_once(exact))
    return labels


def _core(spec: dict[str, Any]) -> dict[str, Any]:
    r"""The case's outcome on the core route, in the recorder's shape."""
    connection = duckdb.connect()
    for statement in spec['setup']:
        connection.execute(statement)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            schema = Schema.from_source(DuckDBSource(connection))
            request = PredictiveQuery(spec['query'], schema).materialize(
                indices=spec['indices'],
                anchor_time=pd.Timestamp(spec['anchor']),
            )
    except Exception as error:  # noqa: BLE001 - the outcome is compared
        return {
            'outcome': 'refused',
            'class': type(error).__name__,
            'code': getattr(error, 'code', None),
            'message': str(error),
            'rule': (getattr(error, 'details', None) or {}).get('rule'),
        }
    payload = request._payloads[0]
    table = payload['context']['instance_table']
    return {
        'outcome': 'labels',
        'task': payload['task'],
        'columns': table['columns'],
        'rows': json.loads(json.dumps(table['rows'], default=str)),
    }


def test_label_fixtures_cover_all_task_kinds() -> None:
    # The vectors came from the reference by the committed recorder, and
    # every outcome the exits name is among them: labels of each kind and
    # the reference's own refusals.
    kinds = {
        case['expected']['task']['kind']
        for case in RECORDED['cases']
        if case['expected']['outcome'] == 'labels'
    }
    assert kinds == {
        'regression',
        'binary_classification',
        'multiclass_classification',
    }
    assert all(
        case['expected']['rows']
        for case in RECORDED['cases']
        if case['expected']['outcome'] == 'labels'
    )


@pytest.mark.parametrize(
    'spec', RECORDED['cases'], ids=[case['name'] for case in RECORDED['cases']]
)
def test_labels_match_expected_values_and_exact_decimal_improvements(
    spec: dict[str, Any],
) -> None:
    got = _core(spec)
    expected = spec['expected']
    if expected['outcome'] == 'refused':
        assert {k: got.get(k) for k in ('outcome', 'class', 'code')} == (
            expected
        ), got.get('message')
    elif got['outcome'] == 'refused':
        pytest.fail(f'refused {got["rule"]!r} where the reference labels')
    elif spec['name'] in EXACT_DECIMAL:
        assert {k: got[k] for k in ('outcome', 'task', 'columns')} == {
            k: expected[k] for k in ('outcome', 'task', 'columns')
        }
        assert [row[:3] for row in got['rows']] == [
            row[:3] for row in expected['rows']
        ]
        exact = _exact_labels(spec, expected['rows'])
        assert [row[3] for row in got['rows']] == exact, spec['covers']
        # Not vacuous: the reference's float64 aggregate loses what the
        # exact one keeps on some row (1e15 + 0.1 - 1e15), and agrees on
        # another.
        reference = [row[3] for row in expected['rows']]
        assert any(r != e for r, e in zip(reference, exact, strict=True))
        assert any(r == e for r, e in zip(reference, exact, strict=True))
    elif spec['name'] in EXACT_CLASSES:
        column = EXACT_CLASSES[spec['name']]
        connection = duckdb.connect()
        for statement in spec['setup']:
            connection.execute(statement)
        by_entity = dict(
            connection.execute(
                f'SELECT user_id, {column} FROM users'
            ).fetchall()
        )
        distinct = {str(v) for v in by_entity.values() if v is not None}
        assert got['outcome'] == expected['outcome'] == 'labels'
        assert got['task']['kind'] == expected['task']['kind']
        assert got['columns'] == expected['columns']
        # One class per distinct exact value, each once.
        classes = got['task']['target']['classes']
        assert len(classes) == len(set(classes))
        assert set(classes) == distinct, classes
        # Each context row: the reference's instance, entity and anchor,
        # labelled with its entity's exact value.
        assert [row[:3] for row in got['rows']] == [
            row[:3] for row in expected['rows']
        ]
        for _, entity, _, label in got['rows']:
            assert label == str(by_entity[entity]), (entity, label)
        # Not vacuous: the reference's float reading merges distinct values
        # (1.5 and 1.500000000000000001) into one class.
        assert len(expected['task']['target']['classes']) < len(distinct)
    else:
        assert got == expected, spec['covers']
