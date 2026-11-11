# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Labels match independently calculated source values through the Python binding."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow as pa
import pytest
from sdm_connectors import DuckDBSource

from sdm_client import PredictiveQuery, Schema
from sdm_client._rust import materialize as bridge

CONTRACT = Path(__file__).resolve().parents[3] / 'contract'
CASES = (
    'base.regression',
    'base.binary',
    'base.multiclass',
    'decimal.target-38-18',
)


def _base(name: str) -> list[str]:
    r"""The statements of base ``name`` of ``contract/bases.yaml``, read
    without a YAML library, which this floor environment lacks: each base
    is a key under ``bases:`` and a list of one-line statements; anything
    else is refused rather than guessed at."""
    lines = (CONTRACT / 'bases.yaml').read_text(encoding='utf-8').splitlines()
    statements: list[str] | None = None
    for line in lines:
        if (
            line.startswith('  ')
            and not line.startswith('  -')
            and line.strip().endswith(':')
        ):
            if statements is not None:
                break
            if line.strip() == f'{name}:':
                statements = []
            continue
        if statements is not None and line.startswith('  - '):
            statements.append(line[4:])
        elif (
            statements is not None
            and line.strip()
            and not line.lstrip().startswith('#')
        ):
            raise AssertionError(
                f'bases.yaml: a line this reader does not take: {line!r}'
            )
    if not statements:
        raise AssertionError(f'bases.yaml: no base {name!r}')
    return statements


def _iso(ns: int) -> str:
    return pd.Timestamp(ns, unit='ns').strftime('%Y-%m-%dT%H:%M:%S.%fZ')


def _core_examples(
    case: dict[str, Any], monkeypatch: Any
) -> list[tuple[Any, str, Any]]:
    r"""The core's context examples for ``case``, on this host: each its
    entity, its anchor as the payload writes it and its label."""
    given = case['input']
    connection = duckdb.connect()
    base = _base(given['base']) if given.get('base') else []
    for statement in base + list(given.get('setup', [])):
        connection.execute(statement)
    labelled: dict[str, Any] = {}
    real = bridge.run

    def labels_in_place(
        operation: dict[str, Any], attachments: Any, **kw: Any
    ) -> Any:
        assert operation['op'] == 'materialize'
        materialize = {k: v for k, v in operation.items() if k != 'op'}
        finished = real(
            {'op': 'test', 'test': 'labels', 'materialize': materialize},
            attachments,
            provider=kw['provider'],
        )
        stream = finished.take(finished.result['data'])
        labelled['table'] = pa.RecordBatchReader._import_from_c_capsule(
            stream.__arrow_c_stream__()
        ).read_all()
        labelled['rows'] = finished.result['rows']
        raise _Labelled

    monkeypatch.setattr(bridge, 'run', labels_in_place)
    schema = Schema.from_source(DuckDBSource(connection))
    arguments: dict[str, Any] = {'indices': given['materialize']['indices']}
    if 'anchor_time' in given['materialize']:
        arguments['anchor_time'] = pd.Timestamp(
            given['materialize']['anchor_time']
        )
    with pytest.raises(_Labelled):
        PredictiveQuery(given['pql'], schema).materialize(**arguments)
    table = labelled['table']
    assert table.num_rows == labelled['rows']
    out = []
    for entity, anchor, target in zip(
        table['entity'].to_pylist(),
        table['anchor'].cast(pa.int64()).to_pylist(),
        table['target'].to_pylist(),
        strict=True,
    ):
        if pa.types.is_float32(table.schema.field('target').type):
            target = float(target)
        out.append((entity, _iso(anchor), target))
    return out


class _Labelled(Exception):
    r"""The labels were taken; the call ends here."""


def _case(name: str) -> dict[str, Any]:
    return json.loads(
        (CONTRACT / f'cases/s1/{name}.json').read_text(encoding='utf-8')
    )


#: Decimal sums whose running total passes 38 digits (or ``i128``) and
#: comes back (fix round 3): user 1's prices in time order and the
#: label the reference gives user 1 at 2024-02-01, recorded from the
#: reference checkout's environment through the public calls (every other
#: example's label 0.0); the client's ``tests/test_the_materialize_route.py``
#: holds the old route to the same rows where the engine is installed.
SIX = '60000000000000000000'
DECIMAL_SUMS = {
    'running-past-38-digits': (
        'DECIMAL(38, 18)',
        [SIX, SIX, '-' + SIX],
        6.000000208206194e19,
    ),
    'back-to-zero': ('DECIMAL(38, 18)', [SIX, SIX, '-' + SIX, '-' + SIX], 0.0),
    'another-order': (
        'DECIMAL(38, 18)',
        [SIX, '-' + SIX, SIX],
        6.000000208206194e19,
    ),
    'scale-0': (
        'DECIMAL(38, 0)',
        ['6' + '0' * 37] * 2 + ['-6' + '0' * 37],
        6.000000213819608e37,
    ),
    'running-past-i128': (
        'DECIMAL(38, 0)',
        ['9' + '0' * 37] * 2 + ['-9' + '0' * 37],
        8.999999813669171e37,
    ),
}
DECIMAL_SUM_EXAMPLES = [
    (0, '2024-02-01'),
    (1, '2024-02-01'),
    (2, '2024-02-01'),
    (3, '2024-02-01'),
    (0, '2024-01-01'),
    (1, '2024-01-01'),
    (3, '2024-01-01'),
    (2, '2024-01-01'),
]


@pytest.mark.parametrize('name', sorted(DECIMAL_SUMS))
def test_a_decimal_sum_s_labels_are_the_reference_s(
    name: str, monkeypatch: Any
) -> None:
    dtype, prices, label = DECIMAL_SUMS[name]
    setup = [
        'CREATE TABLE users (user_id BIGINT PRIMARY KEY)',
        'INSERT INTO users SELECT i FROM range(4) t(i)',
        'CREATE TABLE orders (order_id BIGINT PRIMARY KEY, user_id BIGINT, '
        f'price {dtype}, ts TIMESTAMP)',
        "INSERT INTO orders VALUES (0, 2, 1, '2024-01-01')",
    ] + [
        f'INSERT INTO orders VALUES ({i + 1}, 1, {price}, '
        f"TIMESTAMP '2024-02-10' + to_days({i}))"
        for i, price in enumerate(prices)
    ]
    case = {
        'input': {
            'setup': setup,
            'pql': 'PREDICT SUM(orders.price, 0, 1, months) FOR EACH users.user_id',
            'materialize': {'indices': [1], 'anchor_time': '2024-03-01'},
        }
    }
    wanted = [
        (
            entity,
            f'{day}T00:00:00.000000Z',
            label if (entity, day) == (1, '2024-02-01') else 0.0,
        )
        for entity, day in DECIMAL_SUM_EXAMPLES
    ]
    assert _core_examples(case, monkeypatch) == wanted


@pytest.mark.parametrize('name', CASES)
def test_labels_follow_authored_source_arithmetic(
    name: str, monkeypatch: Any
) -> None:
    actual = _core_examples(_case(name), monkeypatch)
    assert [row[1] for row in actual] == sorted(
        (row[1] for row in actual), reverse=True
    )
    for start in range(0, len(actual), 8):
        window = actual[start : start + 8]
        assert len({row[1] for row in window}) == 1
        assert {row[0] for row in window} == set(range(8))
    if name == 'base.multiclass':
        expected = [
            (
                entity,
                '2024-07-07T00:00:00.000000Z',
                ['US', 'DE', 'IN'][entity % 3],
            )
            for entity in range(8)
        ]
    else:
        expected = []
        for entity in range(8):
            for n in range(14):
                day = 395 - 30 * n
                orders = [
                    i
                    for i in range(80)
                    if i % 8 == entity and day < i * 7 <= day + 30
                ]
                target = (
                    bool(orders)
                    if name == 'base.binary'
                    else sum((10.0 + 1.5 * i for i in orders), 0.0)
                )
                anchor = pd.Timestamp('2023-01-01') + pd.Timedelta(days=day)
                expected.append(
                    (entity, anchor.strftime('%Y-%m-%dT%H:%M:%S.%fZ'), target)
                )
    assert len(actual) == len(expected)
    assert sorted(actual) == sorted(expected)
    assert all(
        type(a[2]) is type(b[2])
        for a, b in zip(sorted(actual), sorted(expected), strict=True)
    )
