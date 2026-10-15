# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The labels on the Python host (T1.27, decision E19).

Each baseline case's context labels, as the core computes them over this
host's fetches, are the reference's exactly, as its payload's context
carries them in the committed reference observation: ``base.regression``
and ``decimal.target-38-18`` (windowed sums at float32), ``base.binary``
(a condition over a count) and ``base.multiclass`` (a static text target).
The case runs through the public calls on the core route, and the
``materialize`` the host starts is run as the test-only ``labels``
operation in its place, with the same inputs and provider, which finishes
with the context's examples. Each reference row is held, by its instance
id, to the core's example of that position: the entity, the anchor to the
microsecond the payload writes, and the label, compared exactly. The
reference leaves out of a static target's context the entities it is asked
to predict for (T1.28a's rule): an example missing from its rows must be
one of those, and the count must be the payload's
``num_context_examples``. ``sdm-testkit``'s ``tests/labels.rs`` holds the
pure-Rust host to the same rows. The check fails closed on missing, empty
or retyped evidence, seeded below.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow as pa
import pytest
from sdm_connectors import DuckDBSource

from sdm_client import PredictiveQuery, Schema
from sdm_client._routing import FROM_SOURCE_DUCKDB, MATERIALIZE_DUCKDB
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


def _reference(case: str) -> Any:
    for path in sorted((CONTRACT / 'reference').glob('*.jsonl')):
        for line in path.read_text(encoding='utf-8').splitlines():
            observed = json.loads(line)
            if observed['case'] == case:
                return observed
    return None


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
    monkeypatch.setenv(
        'SDM_CORE_ROUTES', ','.join(FROM_SOURCE_DUCKDB + MATERIALIZE_DUCKDB)
    )
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


def _cell(value: Any) -> Any:
    if isinstance(value, dict) and set(value) == {'number'}:
        text = value['number']
        try:
            return int(text)
        except ValueError:
            return float(text)
    return value


def problems(
    observed: Any, indices: list[Any], core: list[tuple[Any, str, Any]]
) -> list[str]:
    r"""Each way ``core``'s examples depart from the reference's context in
    ``observed``, for a case asking for ``indices``; missing, empty or
    retyped evidence is a problem."""
    try:
        payload = observed['public']['payloads'][0]['parsed']
        table = payload['context']['instance_table']
        columns = table['columns']
        rows = table['rows']
        count = _cell(payload['metadata']['num_context_examples'])
        at = [
            columns.index(name)
            for name in (
                'instance_id',
                '__kumo_entity_ref_0',
                '__kumo_anchor_time',
                'TARGET',
            )
        ]
    except (KeyError, IndexError, TypeError, ValueError):
        return ['the reference holds no context']
    if not rows:
        return ["the reference's context holds no row"]
    found = []
    if not core:
        found.append('the core gave no label')
    if len(core) != count:
        found.append(
            f'the core gave {len(core)} examples, the reference {count}'
        )
    seen = [False] * len(core)
    for row in rows:
        position = _cell(row[at[0]])
        if not isinstance(position, int) or not 0 <= position < len(core):
            found.append(f'a reference row names no example: {row}')
            continue
        seen[position] = True
        wanted = (_cell(row[at[1]]), row[at[2]], _cell(row[at[3]]))
        mine = core[position]
        # Exactly, and of the same type: a boolean is no integer here.
        if mine != wanted or [type(v) for v in mine] != [
            type(v) for v in wanted
        ]:
            found.append(
                f"example {position}: the core's {mine}, the reference's {wanted}"
            )
    for position, example in enumerate(core):
        if not seen[position] and example[0] not in indices:
            found.append(
                f"example {position} is not the reference's and not one predicted for"
            )
    return found


def _case(name: str) -> dict[str, Any]:
    return json.loads(
        (CONTRACT / f'cases/s1/{name}.json').read_text(encoding='utf-8')
    )


@pytest.mark.parametrize('name', CASES)
def test_the_labels_on_this_host_are_the_reference_s(
    name: str, monkeypatch: Any
) -> None:
    case = _case(name)
    core = _core_examples(case, monkeypatch)
    observed = _reference(name)
    assert (
        problems(observed, case['input']['materialize']['indices'], core) == []
    )


def test_the_label_check_fails_closed(monkeypatch: Any) -> None:
    name = 'base.binary'
    case = _case(name)
    core = _core_examples(case, monkeypatch)
    observed = _reference(name)
    indices = case['input']['materialize']['indices']
    assert problems(observed, indices, core) == []
    flipped = list(core)
    flipped[3] = (core[3][0], core[3][1], not core[3][2])
    assert problems(observed, indices, flipped)
    moved = list(core)
    moved[0] = (core[0][0], '2024-01-30T00:00:00.000000Z', core[0][2])
    assert problems(observed, indices, moved)
    retyped = list(core)
    retyped[1] = (core[1][0], core[1][1], int(core[1][2]))
    assert problems(observed, indices, retyped)
    assert problems(observed, indices, core[1:])
    assert problems(observed, indices, [])
    assert problems(None, indices, core)
    emptied = json.loads(json.dumps(observed))
    emptied['public']['payloads'][0]['parsed']['context']['instance_table'][
        'rows'
    ] = []
    assert problems(emptied, indices, core)
    left_out = json.loads(json.dumps(observed))
    del left_out['public']['payloads'][0]['parsed']['context'][
        'instance_table'
    ]['rows'][0]
    assert problems(left_out, [], core)
    assert os.environ.get('SDM_CORE_ROUTES')


#: Decimal sums whose running total passes 38 digits (or ``i128``) and
#: comes back (T1.27's fix round 3): user 1's prices in time order and the
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
