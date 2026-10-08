# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The core's cast of ``indices`` to a ``decimal128`` key, against today's
``_get_task_table``, over generated lists.

``_get_task_table`` casts the caller's ``indices`` with
``pd.Series(indices, dtype=<the key's dtype>)``: what pandas and pyarrow
make of a list mixing ``None``, bools, ints, floats and texts depends on
every value and on their order. ``duckdb/regression#decimal-entity-keys``
asks for products whose ids are ``DECIMAL(10,0)``. Its ``_get_task_table``
call is kept as the transcript replays, then made again with each generated
list as ``indices``, the database answered from the kept statements; the
core's ``task_table`` run is driven from each call's inputs and must give
what today's call gave, or raise the same class with the same text, as
``test_rfm_runs.py`` compares them.

The key is the recorded ``decimal128(10, 0)``, and also ``(38, 0)``,
``(19, 0)`` and ``(12, 2)``, with the kept results' decimal columns cast to
them: ints past ``int64`` go straight to a key wide enough for them, and the
scale changes what fits.

The core's inputs are JSON written with ``allow_nan=False``, and
``capture_sampling.py`` refuses an index JSON does not hold, so a list
holding a ``decimal.Decimal`` or ``NaN`` never reaches the core; the test
checks that it is refused there.
"""

from __future__ import annotations

import contextlib
import itertools
import math
import re
import sys
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip('sdm_core')
pytest.importorskip('sdm_relational_engine')

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Today's Python, which these hold the core to, runs public calls here.
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _old_path import python_path  # noqa: E402,F401 - the fixture

pytestmark = pytest.mark.usefixtures('python_path')

from test_rfm_runs import (  # noqa: E402
    REPO,
    Boundary,
    _compare,
    _model_describer,
    _source_classes,
    task_table_got,
    task_table_want,
)

SCENARIO = (
    REPO
    / 'tests'
    / 'parity'
    / 'transcripts'
    / 'duckdb'
    / 'regression--decimal-entity-keys'
    / 'transcript.json'
)

#: Values JSON holds, one of each kind and edge pyarrow treats apart: ints
#: in and past ``int64`` and past what a double holds exactly, floats that
#: round, fit or do not, and texts that cast or do not.
VALUES: list[Any] = [
    None,
    True,
    False,
    0,
    1,
    -1,
    2**53 + 1,
    -(2**63),
    2**63,
    2**64 - 1,
    12_345_678_901,
    1.5,
    2.5,
    -0.0,
    1e20,
    '1',
    '1.5',
    '1e2',
    'x',
]

#: One value of each kind, for longer lists.
KINDS: list[Any] = [None, True, 1, 2**63, 1.5, '1', 'x']

#: Values the core's inputs cannot hold.
UNHELD: list[Any] = [Decimal('1'), Decimal('1.5'), math.nan]


def _lists() -> list[list[Any]]:
    lists: list[list[Any]] = [[]]
    for length in (1, 2):
        lists += [list(c) for c in itertools.product(VALUES, repeat=length)]
    lists += [list(c) for c in itertools.product(KINDS, repeat=3)]
    lists += [
        list(c) for c in itertools.product([None, True, 1, 1.5, '1'], repeat=4)
    ]
    return lists


def _unheld_lists() -> list[list[Any]]:
    return [
        [*before, value, *after]
        for value in UNHELD
        for before, after in (([], []), ([1], []), ([], [1.5]), ([None], []))
    ]


def _kept_call() -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    r"""The scenario's ``_get_task_table`` call for products: the model, its
    arguments and the statements it ran."""
    from _parity.corpus import check
    from sdm_relational_engine.rfm.rfm import KumoRelational

    kept: list[tuple[Any, dict[str, Any]]] = []
    boundary = Boundary(
        KumoRelational, '_get_task_table', lambda model: model._sampler
    )

    def describe(model: Any, arguments: dict[str, Any]) -> Any:
        kept.append((model, dict(arguments)))
        return _model_describer('_get_task_table')(model, arguments)

    with boundary.installed(describe):
        reason = check(SCENARIO)
    from _fixed_contracts import fixed_replay

    fixed_replay(SCENARIO, reason)
    found = [
        (model, arguments, call['steps'])
        for (model, arguments), call in zip(kept, boundary.calls, strict=True)
        if arguments['query'].entity_table == 'products'
    ]
    assert len(found) == 1, f'{len(found)} calls for products'
    return found[0]


def _retyped(result: Any, key_type: tuple[int, int] | None) -> Any:
    r"""A kept result with its decimal columns cast to ``key_type``."""
    import pyarrow as pa

    if key_type is None or not isinstance(result, pa.Table):
        return result
    schema = pa.schema(
        field.with_type(pa.decimal128(*key_type))
        if pa.types.is_decimal(field.type)
        else field
        for field in result.schema
    )
    return result.cast(schema)


@contextlib.contextmanager
def _answering(
    steps: list[dict[str, Any]], key_type: tuple[int, int] | None
) -> Iterator[None]:
    r"""Every ``Source._run`` answered from the kept statement of the same
    text, its decimal columns cast to ``key_type`` where one is given.

    By text, not in order: the engine keeps some results on the sampler
    after the first call, so later calls ask fewer statements. The text is
    matched with any ``DECIMAL(p, s)`` in it set aside, and the parameters
    are not matched: both name or hold keys, whose type ``key_type``
    changes. The core must still ask exactly what today's call asked.
    """

    def text(sql: str) -> str:
        return re.sub(r'DECIMAL\(\d+, \d+\)', 'DECIMAL', sql)

    kept = {text(step['sql']): step for step in steps}
    assert len(kept) == len(steps), 'a statement was asked twice'

    def run(source: Any, sql: Any, params: Any = None) -> Any:
        step = kept.get(text(sql))
        assert step is not None, f'a statement that was not kept: {sql}'
        assert step['error'] is None, step['error']
        return _retyped(step['result'], key_type)

    undo = []
    for cls in _source_classes():
        undo.append((cls, vars(cls)['_run']))
        cls._run = run
    try:
        yield
    finally:
        for cls, original in undo:
            cls._run = original


@pytest.mark.parametrize(
    'key_type',
    [None, (38, 0), (19, 0), (12, 2)],
    ids=['as-recorded', 'decimal-38-0', 'decimal-19-0', 'decimal-12-2'],
)
def test_decimal_indices(key_type: tuple[int, int] | None) -> None:
    r"""``key_type``: the key's ``decimal128`` precision and scale, where
    not the recorded ``(10, 0)``."""
    from sdm_relational_engine.rfm.rfm import KumoRelational

    model, arguments, steps = _kept_call()
    lists = _lists()
    boundary = Boundary(
        KumoRelational, '_get_task_table', lambda model: model._sampler
    )
    refused_before = []
    with _answering(steps, key_type):
        with boundary.installed(_model_describer('_get_task_table')):
            for indices, held in [
                *((given, True) for given in lists),
                *((given, False) for given in _unheld_lists()),
            ]:
                calls = len(boundary.calls)
                try:
                    model._get_task_table(**{**arguments, 'indices': indices})
                except ValueError as error:
                    if not held:
                        refused_before.append(str(error))
                except Exception:  # noqa: BLE001, S110 - compared below
                    pass
                assert len(boundary.calls) == calls + held, indices
    assert len(refused_before) == len(_unheld_lists()), refused_before
    assert all(
        'that JSON does not hold' in text or 'not JSON compliant' in text
        for text in refused_before
    ), refused_before
    count, problems = _compare(
        'task_table', boundary, task_table_want, task_table_got
    )
    assert count == len(lists)
    outcomes = {
        'finished': sum(call['error'] is None for call in boundary.calls),
        **{
            name: sum(
                type(call['error']).__name__ == name for call in boundary.calls
            )
            for name in (
                'ValueError',
                'RuntimeError',
                'ArrowNotImplementedError',
                'OverflowError',
            )
        },
    }
    key = 'decimal128({}, {})[pyarrow]'.format(*(key_type or (10, 0)))
    named = [
        str(call['error'])
        for call in boundary.calls
        if type(call['error']).__name__ == 'RuntimeError'
        or str(call['error']).startswith("'indices' must hold")
    ]
    assert named, 'no error names the key'
    assert all(key in text for text in named), f'the key is not {key}'
    # Every list ends in an error: a decimal entity key is refused once the
    # task table is built (the scenario's RuntimeError).
    assert outcomes['finished'] == 0, outcomes
    assert all(outcomes[name] for name in outcomes if name != 'finished'), (
        outcomes
    )
    assert not problems, '\n'.join(
        f'{lists[int(p.split()[1].rstrip(":"))]!r}: {p}' for p in problems[:20]
    )
