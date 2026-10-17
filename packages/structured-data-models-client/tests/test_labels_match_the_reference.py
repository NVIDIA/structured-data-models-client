# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The core route's labels are the reference's, compared exactly (T1.27).

Written by the package's independent test author from the brief and the
reference, not from the implementation. Each case in
``labels_reference.json`` is a DuckDB source, a query and an anchor time;
``labels_reference_record.py`` recorded what the reference's
``PredictiveQuery.materialize`` gives for it: the task and the context
examples (instance, entity, anchor time, label), or the refusal's class and
code. Here the same call runs forced onto the core
(``SDM_CORE_ROUTES``, the schema's route with it) and must give the same,
exactly: float32 labels compared as the numbers they are, not within a
tolerance (E19).

What the cases cover, each case naming its line in ``covers``:

- the exit "DateOffset window ends at month ends and leap days match the
  reference's ``_by_time`` seeds": month windows from 2024-03-31 (a context
  anchor at 2024-02-29, its window ending 2024-03-29) and from 2023-03-31,
  Jan 31 plus a month, twelve months back from 2025-02-28, a window that
  starts after its anchor, an order at the anchor and a microsecond after
  the window's end;
- the exit "labels in ``base.regression``'s payload ... compared exactly":
  SUM, COUNT, AVG, MIN and MAX over a float, an integer and a
  DECIMAL(38,18) column, float32 sums where Kahan's compensation decides the
  label (2**24 plus ones, twenty 0.1s, 1e30 plus small values), a decimal
  sum a float64 sum would not keep (1e15 + 0.1 - 1e15), null values, an
  all-null column, windows with no rows;
- the exit "binary and multiclass labels match the reference exactly": a
  condition over a count with each operator, equality with a float
  constant read at float32, an integer target against a float constant, a
  decimal sum against a constant, and static targets (a VARCHAR column of
  twelve classes, a BOOLEAN, a DOUBLE, a condition on a static column, and
  the decimal class target of ``decimal.class-target-38-18``).

A case the core refuses by a rule ``OWNED`` names (the classification and
static tasks' layout, a decimal feature on the wire, a static decimal
target) is held to that refusal having an owner the register names, the
exit "the next refusal each probe hits has an owner", and becomes an exact
comparison once its owner lifts it; a windowed regression over a float or
an integer column may never be refused.

Since main's eb6c68eb (the reference at 6298913c) the reference sums and
averages a DECIMAL(38,18) column as float64, where the core keeps exact
decimals (register P2B-1, until T2.17): on ``sum-decimal`` and
``avg-decimal`` the core is held to the exact aggregate DuckDB computes
over the case's own rows, read at float32, and to the reference in
everything else.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pytest
from sdm_connectors import DuckDBSource

from sdm_client import PredictiveQuery, Schema, _routing

ROOT = Path(__file__).resolve().parents[3]
RECORDED = json.loads(
    Path(__file__)
    .with_name('labels_reference.json')
    .read_text(encoding='utf-8')
)
CORE = ','.join(_routing.FROM_SOURCE_DUCKDB + _routing.MATERIALIZE_DUCKDB)

# The refusals the core still makes, by name, after computing the labels:
# the classification and static tasks' layout in the payload and a decimal
# feature on the wire are T1.28a's, a static decimal target P1B-18's
# (register P1B-3). A case refused so is held to the refusal's owner, not
# to the reference's labels, until the owner lifts it; then it is
# compared exactly like the rest.
OWNED = {
    'classification_task': 'T1.28a',
    'wire_dtype': 'T1.28a',
    'target_leakage': 'T1.28a',
    'target_type': 'P1B-18',
}


#: Register P2B-1 (until T2.17): since main's eb6c68eb the reference sums
#: and averages a DECIMAL column as float64, where the core keeps Phase 1's
#: exact decimals. On these cases the core's labels are held to the exact
#: aggregate DuckDB computes over the case's own rows, read at float32,
#: and everything else to the reference's.
EXACT_DECIMAL = {'sum-decimal': 'SUM', 'avg-decimal': 'AVG'}


def _exact_labels(spec: dict[str, Any], rows: list[list[Any]]) -> list[float]:
    r"""Each context row's label as the exact decimal aggregate of its
    entity's orders in the window after its anchor, at float32."""
    import numpy as np

    aggregate = EXACT_DECIMAL[spec['name']]
    assert spec['query'] == (
        f'PREDICT {aggregate}(orders.amount, 0, 30, days) FOR EACH users.user_id'
    )
    connection = duckdb.connect()
    for statement in spec['setup']:
        connection.execute(statement)
    labels = []
    for _, entity, anchor, _ in rows:
        (value,) = connection.execute(
            f'SELECT {aggregate}(amount) FROM orders WHERE user_id = ? '
            'AND ts > CAST(? AS TIMESTAMP) '
            'AND ts <= CAST(? AS TIMESTAMP) + INTERVAL 30 DAY',
            [entity, anchor[:19], anchor[:19]],
        ).fetchone()
        labels.append(float(np.float32(value)))
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


def test_the_recording_is_the_reference_s() -> None:
    # The vectors came from the reference by the committed recorder, and
    # every outcome the exits name is among them: labels of each kind and
    # the reference's own refusals.
    assert RECORDED['recorded_by'] == 'labels_reference_record.py'
    assert len(RECORDED['reference']) == 40
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
def test_the_core_route_s_labels_are_the_reference_s(
    monkeypatch: pytest.MonkeyPatch, spec: dict[str, Any]
) -> None:
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    got = _core(spec)
    expected = spec['expected']
    if got['outcome'] == 'refused' and got['rule'] in OWNED:
        # The next refusal the case hits has an owner (the exit), and the
        # register names the rule and its owner.
        assert got['code'] == 'UNSUPPORTED_REQUEST', got['message']
        register = (ROOT / 'docs/design/rust-core/register.yaml').read_text(
            encoding='utf-8'
        )
        assert got['rule'] in register and OWNED[got['rule']] in register
        # Never a windowed regression over a float or an integer: those
        # labels are the core's from T1.27, and their payload too.
        assert (
            expected['outcome'] == 'refused'
            or expected['task']['kind'] != 'regression'
            or 'orders.amount' in spec['query']
            or spec['query'].startswith('PREDICT users.')
        ), got['rule']
        return
    if expected['outcome'] == 'refused':
        assert {k: got.get(k) for k in ('outcome', 'class', 'code')} == (
            expected
        ), got.get('message')
    elif got['outcome'] == 'refused':
        pytest.fail(f'refused {got["rule"]!r} where the reference labels')
    elif spec['name'] in EXACT_DECIMAL:
        register = (ROOT / 'docs/design/rust-core/register.yaml').read_text(
            encoding='utf-8'
        )
        assert '- id: P2B-1\n' in register
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
    else:
        assert got == expected, spec['covers']
