# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""``diff.py --self-test``: seeded single changes, each found in place.

The inputs are made here, not read from the corpus, so the test means the
same thing whatever the corpus holds. Each seed changes one thing in a copy
and names the path the difference must be reported at; identical inputs, and
the same inputs laid out or keyed differently, must report nothing.
"""

from __future__ import annotations

import copy
import json
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pyarrow as pa

from _parity.diffing import compare_files, render
from _parity.transcript import write_arrow


def _table(columns: list[str], rows: list[list[Any]]) -> dict[str, Any]:
    return {'columns': columns, 'format': 'arrays', 'rows': rows}


def payloads() -> list[dict[str, Any]]:
    orders = [
        [0, i, 10.0 + i, f'2026-01-{i + 1:02d}T00:00:00.000000Z', i % 3]
        for i in range(12)
    ]
    return [
        {
            'model': 'kumo-tabular',
            'metadata': {'request_id': 'a'},
            'output': {'fields': ['prediction']},
            'task': {
                'kind': 'classification',
                'target': {'column_name': 'churn'},
            },
            'context': {
                'instance_table': _table(
                    ['user_id', 'age', 'churn'],
                    [[i, 20.0 + i, i % 2] for i in range(6)],
                )
            },
        },
        {
            'model': 'kumo-relational',
            'metadata': {'request_id': 'b', 'source_query': 'PREDICT x'},
            'task': {'kind': 'regression'},
            'inference': {'inference_config': {'column_shuffle': False}},
            'schema': {
                'relationships': [
                    {'source_columns': ['user_id'], 'target_table': 'people'},
                    {'source_columns': ['order_id'], 'target_table': 'orders'},
                ],
                'related_tables': {
                    'orders': {
                        'columns': {
                            'price': {'dtype': 'float64', 'stype': 'numerical'}
                        }
                    }
                },
            },
            'context': {
                'instance_table': _table(
                    ['instance_id', 'ref', 'TARGET'],
                    [[i, i * 10, float(i)] for i in range(5)],
                ),
                'related_tables': {
                    'orders': _table(
                        [
                            'instance_id',
                            '__node_id',
                            'price',
                            'ordered_at',
                            'user_id',
                        ],
                        orders,
                    )
                },
            },
        },
    ]


_ORDERS = '[1].context.related_tables.orders'


def _set(path: list[Any], new: Callable[[Any], Any]) -> Callable[[Any], None]:
    def seed(doc: Any) -> None:
        target = doc
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = new(target[path[-1]])

    return seed


def _swap(path: list[Any], a: int, b: int) -> Callable[[Any], None]:
    def seed(doc: Any) -> None:
        target = doc
        for part in path:
            target = target[part]
        target[a], target[b] = target[b], target[a]

    return seed


def _drop(path: list[Any], index: int) -> Callable[[Any], None]:
    def seed(doc: Any) -> None:
        target = doc
        for part in path:
            target = target[part]
        del target[index]

    return seed


_O = [1, 'context', 'related_tables', 'orders']
_I = [1, 'context', 'instance_table']

#: Payload seeds: what to change, and where it must be reported.
PAYLOAD_SEEDS: dict[str, tuple[Callable[[Any], None], str]] = {
    'a float cell': (
        _set([*_O, 'rows', 7, 2], lambda v: v + 0.5),
        f'{_ORDERS}.rows[7].price',
    ),
    'an integer turned float': (
        _set([*_I, 'rows', 1, 1], float),
        '[1].context.instance_table.rows[1].ref',
    ),
    'a cell turned null': (
        _set([*_O, 'rows', 2, 2], lambda v: None),
        f'{_ORDERS}.rows[2].price',
    ),
    'a timestamp': (
        _set([*_O, 'rows', 3, 3], lambda v: v.replace('T00', 'T01')),
        f'{_ORDERS}.rows[3].ordered_at',
    ),
    'a reordered row': (
        _swap([*_O, 'rows'], 5, 6),
        f'{_ORDERS}.rows[5].__node_id',
    ),
    'a dropped row': (
        _drop([*_O, 'rows'], 10),
        f'{_ORDERS}.rows[10].__node_id',
    ),
    'a dropped last row': (
        _drop([*_I, 'rows'], 4),
        '[1].context.instance_table.rows[4]',
    ),
    'a reordered column': (
        _swap([*_O, 'columns'], 0, 1),
        f'{_ORDERS}.columns[0]',
    ),
    'a changed dtype': (
        _set(
            [
                1,
                'schema',
                'related_tables',
                'orders',
                'columns',
                'price',
                'dtype',
            ],
            lambda v: 'float32',
        ),
        '[1].schema.related_tables.orders.columns.price.dtype',
    ),
    'a removed key': (
        _drop(
            [1, 'schema', 'related_tables', 'orders', 'columns', 'price'],
            'stype',
        ),
        '[1].schema.related_tables.orders.columns.price.stype',
    ),
    'an added key': (
        _set([1, 'inference', 'inference_config'], lambda v: {**v, 'extra': 1}),
        '[1].inference.inference_config.extra',
    ),
    'a flipped bool': (
        _set(
            [1, 'inference', 'inference_config', 'column_shuffle'],
            lambda v: not v,
        ),
        '[1].inference.inference_config.column_shuffle',
    ),
    'a bool turned integer': (
        _set([1, 'inference', 'inference_config', 'column_shuffle'], int),
        '[1].inference.inference_config.column_shuffle',
    ),
    'a changed task kind': (
        _set([1, 'task', 'kind'], lambda v: 'classification'),
        '[1].task.kind',
    ),
    'reordered relationships': (
        _swap([1, 'schema', 'relationships'], 0, 1),
        '[1].schema.relationships[0].source_columns[0]',
    ),
    'an extra payload': (lambda doc: doc.append(copy.deepcopy(doc[0])), '[2]'),
    'a first-payload integer turned string': (
        _set([0, 'context', 'instance_table', 'rows', 0, 0], str),
        '[0].context.instance_table.rows[0].user_id',
    ),
}

#: Changes that must not count: request_id, key order, layout.
PAYLOAD_SAME: dict[str, Callable[[Any], None]] = {
    'a changed request_id': _set(
        [1, 'metadata', 'request_id'], lambda v: 'zzz'
    ),
    'reordered keys': lambda doc: doc.__setitem__(
        1, dict(reversed(doc[1].items()))
    ),
}


def _transcript(root: Path) -> Path:
    root.mkdir(parents=True)
    steps = []
    for index, (sql, params) in enumerate(
        [
            ('SELECT 1', None),
            ('SELECT "a" FROM "t" WHERE "k" = ?', [3]),
            ('SELECT "b" FROM "t" WHERE "k" IN (?, ?)', [4, 5]),
        ]
    ):
        name = f'step-{index:04d}.arrow'
        write_arrow(
            root / name, pa.table({'a': [index, index + 1], 'b': ['x', 'y']})
        )
        steps.append(
            {
                'op': 'execute',
                'sql': sql,
                'params': params,
                'result': name,
                'error': None,
            }
        )
    write_arrow(root / 'result.arrow', pa.table({'prediction': [0.25, 0.75]}))
    transcript = {
        'format_version': 1,
        'name': 'self-test',
        'backend': 'sqlite',
        'task_kind': 'count',
        'data': {'generator': 'shop', 'seed': 0},
        'calls': [],
        'steps': steps,
        'payloads': payloads()[1:],
        'result': 'result.arrow',
    }
    path = root / 'transcript.json'
    path.write_text(json.dumps(transcript))
    return path


def _rewrite_result(name: str) -> Callable[[dict[str, Any], Path], None]:
    def seed(transcript: dict[str, Any], root: Path) -> None:
        write_arrow(root / name, pa.table({'a': [7, 8], 'b': ['x', 'y']}))

    return seed


TRANSCRIPT_SEEDS: dict[
    str, tuple[Callable[[dict[str, Any], Path], None], str]
] = {
    'a SQL character': (
        lambda t, r: t['steps'][1].__setitem__(
            'sql', t['steps'][1]['sql'].lower()
        ),
        'steps[1].sql',
    ),
    'a SQL parameter': (
        lambda t, r: t['steps'][2]['params'].__setitem__(1, 6),
        'steps[2].params[1]',
    ),
    'a dropped step': (lambda t, r: t['steps'].pop(1), 'steps[1].sql'),
    'a recorded result': (
        _rewrite_result('step-0002.arrow'),
        'steps[2].result[0].a',
    ),
    'the decoded result': (
        lambda t, r: write_arrow(
            r / 'result.arrow', pa.table({'prediction': [0.25, 0.5]})
        ),
        'result[1].prediction',
    ),
}


def run_self_test() -> int:
    failures = []
    count = 0
    with tempfile.TemporaryDirectory(prefix='sdm-parity-diff-') as scratch:
        base = Path(scratch)
        a = base / 'a.json'
        a.write_text(json.dumps(payloads()))
        for name, (seed, expected) in PAYLOAD_SEEDS.items():
            count += 1
            changed = payloads()
            seed(changed)
            b = base / 'b.json'
            b.write_text(json.dumps(changed, indent=2))
            found = compare_files(a, b)
            got = render(found[0]) if found else None
            if got != expected:
                failures.append(f'{name}: expected {expected}, got {got}')
        for name, seed in PAYLOAD_SAME.items():
            same = payloads()
            seed(same)
            b = base / 'b.json'
            b.write_text(json.dumps(same, indent=3, sort_keys=True))
            if compare_files(a, b) is not None:
                failures.append(f'{name}: reported a difference')
        original = _transcript(base / 'original')
        for name, (seed, expected) in TRANSCRIPT_SEEDS.items():
            count += 1
            copy_root = base / 'seeded'
            if copy_root.exists():
                shutil.rmtree(copy_root)
            shutil.copytree(original.parent, copy_root)
            transcript = json.loads((copy_root / 'transcript.json').read_text())
            seed(transcript, copy_root)
            (copy_root / 'transcript.json').write_text(
                json.dumps(transcript, indent=1)
            )
            found = compare_files(original, copy_root / 'transcript.json')
            got = render(found[0]) if found else None
            if got != expected:
                failures.append(f'{name}: expected {expected}, got {got}')
        elsewhere = base / 'elsewhere'
        shutil.copytree(original.parent, elsewhere)
        if compare_files(original, elsewhere / 'transcript.json') is not None:
            failures.append(
                'a copy of the transcript elsewhere: reported a difference'
            )
    for failure in failures:
        print(f'FAIL {failure}')
    print(f'{count - len(failures)} of {count} seeded changes found in place')
    return 1 if failures else 0
